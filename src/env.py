"""共享的编辑-打分环境。所有 policy（random / round-robin / gap / greedy / bandit / DQN）
都跑在这个类上，这是 paired 比较成立的前提。

设计严格照 proposal：
- 所有合格候选进 archive，archive 取非支配集
- 下一个分子 = 单候选 HV 增益最大者；并列时按 canonical SMILES 字典序
- 无合格候选则保持当前分子
- reward = HV(A_{t+1}) - HV(A_t) - λ·c_t

**transition rule 对每个 policy 完全相同**，policy 唯一能改变的是 action 序列。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .chem import canonical, scaffold_preserved
from .editors import make_editors
from .interfaces import EXPERTS, N_OBJ, OBJECTIVES, Candidate, EpisodeConfig, StepLog
from .oracles import OracleBank

RDLogger.DisableLog("rdApp.*")

_FP_BITS = 64  # 折叠得很小：1024 位会淹没其余特征，DQN 在 200 个 seed 上学不动
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=_FP_BITS)


# --------------------------------------------------------------------------
# Hypervolume
# --------------------------------------------------------------------------
def hypervolume(points: list[tuple[float, ...]], ref: tuple[float, ...]) -> float:
    """最大化方向的 HV。pymoo 按最小化算，所以取 1-f 并把参考点搬到 1-ref。"""
    if not points:
        return 0.0
    from pymoo.indicators.hv import HV

    F = 1.0 - np.asarray(points, dtype=float)
    ref_min = 1.0 - np.asarray(ref, dtype=float)
    # 落在参考点之外的点对 HV 无贡献，pymoo 会自行处理
    return float(HV(ref_point=ref_min)(F))


def nondominated(points: list[tuple[str, tuple[float, ...]]]):
    """返回非支配子集。点数很小（<100），O(n^2) 足够。"""
    keep = []
    for i, (si, vi) in enumerate(points):
        dominated = False
        for j, (sj, vj) in enumerate(points):
            if i == j:
                continue
            if all(a >= b for a, b in zip(vj, vi)) and any(
                a > b for a, b in zip(vj, vi)
            ):
                dominated = True
                break
        if not dominated:
            keep.append((si, vi))
    return keep


def _fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(_FP_BITS, dtype=np.float32)
    return np.asarray(_fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
@dataclass
class MolEnv:
    """一个 episode = 一个 seed 分子上的一条轨迹。"""

    bank: OracleBank
    cfg: EpisodeConfig = field(default_factory=EpisodeConfig)
    use_fingerprint: bool = True
    # 诊断（scripts/analyze_editor_depletion.py）发现的吸收态修复：
    # 某步产出 0 个合格候选 -> 分子不动 -> 确定性 policy 看到同样的 obs
    # -> 重选同一个 editor -> 又是 0，永远卡住。gap 有 45% 的步是这种空转。
    # 这四位记录"哪些 editor 已经在当前分子上失败过"，分子一变就清空。
    # 默认关闭：打开会改变 obs_dim，旧的 policies.pkl 不兼容。
    use_escape_bits: bool = False
    # editor 池：'rdkit' 是确定性反应池（proposal 把它定为 supplementary
    # control），'gen' 是条件 SMILES transformer（proposal §3.1 的主实验）。
    # 见 docs/findings.md §9。两者都遵守同一个 Editor 契约，env 其余部分不变。
    editor_kind: str = "rdkit"
    gen_ckpt: str = "results_gen/generator.pt"
    # 只对 editor_kind="rdkit" 有意义：人为调节四个池的重叠程度，
    # 用来把"routing 收益 vs action 空间维度"从两个点做成一条曲线。
    # 见 src/editors.py:_mix 和 docs/findings.md §11c。默认值 = 历史行为。
    pool_overlap: float = 0.0
    qed_disjoint: bool = False

    def __post_init__(self) -> None:
        if self.editor_kind == "gen":
            from .gen_editors import make_gen_editors
            # 共用 env 自己的 bank：当前分子的分数已经算过，走 cache 不额外计费
            self.editors = make_gen_editors(self.gen_ckpt, self.bank)
        elif self.editor_kind == "rdkit":
            self.editors = make_editors(overlap=self.pool_overlap,
                                        qed_disjoint=self.qed_disjoint)
        else:
            raise ValueError(f"未知 editor_kind: {self.editor_kind}")
        self.reset_counters()

    def reset_counters(self) -> None:
        self.n_generation_attempts = 0
        self.n_oracle_calls_at_start = self.bank.n_calls

    # ---- episode lifecycle ----
    def reset(self, seed_smiles: str, episode_seed: int):
        self.seed_smiles = canonical(seed_smiles) or seed_smiles
        self.episode_seed = episode_seed
        self.current = self.seed_smiles
        self.t = 0
        self.prev_action = -1
        self.failed_on_current: set[int] = set()
        self.logs: list[StepLog] = []
        self.seen: set[str] = {self.seed_smiles}
        self.n_generation_attempts = 0

        v = self.bank.vector(self.seed_smiles)
        if v is None:
            raise ValueError(f"seed 分子非法: {seed_smiles}")
        self.archive: list[tuple[str, tuple[float, ...]]] = [
            (self.seed_smiles, tuple(v))
        ]
        self.hv = hypervolume([tuple(v)], self.cfg.ref_point)
        self.seed_hv = self.hv
        return self._obs()

    def _obs(self) -> np.ndarray:
        cur = self.bank.vector(self.current) or [0.0] * N_OBJ
        prev = np.zeros(len(EXPERTS) + 1, dtype=np.float32)
        prev[self.prev_action if self.prev_action >= 0 else len(EXPERTS)] = 1.0
        arch = np.asarray([v for _, v in self.archive], dtype=np.float32)
        feats = [
            np.asarray(cur, dtype=np.float32),            # 当前分子的四个目标分
            prev,                                          # 上一个 expert 的 one-hot
            np.asarray([1.0 - self.t / self.cfg.max_steps], dtype=np.float32),
            np.asarray([self.hv], dtype=np.float32),
            arch.max(axis=0),                              # archive 各目标最大值
            np.asarray([len(self.archive) / 20.0], dtype=np.float32),
        ]
        if self.use_escape_bits:
            bits = np.zeros(len(EXPERTS), dtype=np.float32)
            for a in self.failed_on_current:
                bits[a] = 1.0
            feats.append(bits)
        if self.use_fingerprint:
            feats.append(_fingerprint(self.current))
        return np.concatenate(feats)

    @property
    def obs_dim(self) -> int:
        return N_OBJ + len(EXPERTS) + 1 + 1 + 1 + N_OBJ + 1 + (
            len(EXPERTS) if self.use_escape_bits else 0
        ) + (_FP_BITS if self.use_fingerprint else 0)

    # ---- one step ----
    def step(self, action: int):
        expert = EXPERTS[action]
        parent = self.current
        base = self.bank.vector(parent)
        hv_before = self.hv
        oracle_before = self.bank.n_calls

        # 每步的 editor 种子由 (episode, t) 决定：所有 policy 在同一 (分子, expert)
        # 上拿到完全相同的候选，差异只来自 action 序列。
        raw = self.editors[expert].propose(
            parent, self.cfg.k_per_call, seed=self.episode_seed * 1000 + self.t
        )
        self.n_generation_attempts += self.cfg.k_per_call

        cands: list[Candidate] = []
        for smi in raw:
            c = canonical(smi)
            if c is None:
                cands.append(Candidate(None, parent, expert, False, False))
                continue
            ok_scaf = (not self.cfg.enforce_scaffold) or scaffold_preserved(parent, c)
            v = self.bank.vector(c) if ok_scaf else None
            cands.append(
                Candidate(c, parent, expert, True, ok_scaf,
                          tuple(v) if v is not None else None)
            )

        eligible = [c for c in cands if c.eligible and c.scores is not None]
        n_novel = sum(1 for c in eligible if c.smiles not in self.seen)

        # --- archive 更新：所有合格候选都进 ---
        pool = self.archive + [(c.smiles, c.scores) for c in eligible]
        # 去重后取非支配
        dedup: dict[str, tuple[float, ...]] = {}
        for s, v in pool:
            dedup[s] = v
        new_archive = nondominated(sorted(dedup.items()))
        hv_after = hypervolume([v for _, v in new_archive], self.cfg.ref_point)

        # --- transition：单候选 HV 增益最大者，并列按 canonical SMILES 字典序 ---
        best_smi, best_gain = None, -1.0
        for c in sorted(eligible, key=lambda x: x.smiles):
            hv_c = hypervolume(
                [v for _, v in nondominated(
                    sorted({**dedup_base(self.archive), c.smiles: c.scores}.items()))],
                self.cfg.ref_point,
            )
            gain = hv_c - hv_before
            if gain > best_gain + 1e-12:
                best_gain, best_smi = gain, c.smiles

        entered = len(new_archive) != len(self.archive) or any(
            s not in {a for a, _ in self.archive} for s, _ in new_archive
        )

        best_single_gain = 0.0
        if eligible and base is not None:
            best_single_gain = max(
                max(c.scores[j] - base[j] for j in range(N_OBJ)) for c in eligible
            )

        self.archive = new_archive
        self.hv = hv_after
        if best_smi is not None and best_smi != parent:
            # 分子动了：之前的失败记录对新分子无意义，清空
            self.current = best_smi
            self.failed_on_current = set()
        elif not eligible:
            # 这个 editor 在当前分子上颗粒无收，记下来
            self.failed_on_current.add(action)
        for c in eligible:
            self.seen.add(c.smiles)

        # --- reward ---
        if self.cfg.cost_unit == "oracle":
            cost = self.bank.n_calls - oracle_before
        else:
            cost = self.cfg.k_per_call
        reward = (hv_after - hv_before) - self.cfg.lam * cost

        self.logs.append(StepLog(
            t=self.t, expert=expert, current=parent,
            n_proposed=self.cfg.k_per_call,
            n_valid=sum(1 for c in cands if c.valid),
            n_scaffold_ok=sum(1 for c in cands if c.eligible),
            n_novel=n_novel,
            hv_before=hv_before, hv_after=hv_after, reward=reward,
            archive_size=len(self.archive), entered_archive=entered,
            best_single_gain=best_single_gain,
        ))

        self.prev_action = action
        self.t += 1
        done = self.t >= self.cfg.max_steps
        return self._obs(), reward, done, {
            "hv": hv_after,
            "hv_gain_vs_seed": hv_after - self.seed_hv,
            "generation_attempts": self.n_generation_attempts,
        }


def dedup_base(archive):
    return {s: v for s, v in archive}
