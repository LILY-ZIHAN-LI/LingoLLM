"""生成式 objective-conditioned editor：共享一个 backbone，四套条件。

这是 proposal §3.1 真正要的东西，RDKit 池只是它的 supplementary control
（见 docs/findings.md §9）。和 RDKit 池的本质区别：

  RDKit：action = 反应模板集合。qed/sa 在 70/100 个种子上**字面共用同一个池**，
         所以 effective action dim 只有 2.05/4（findings §1b）。
  这里：action = 条件向量。四个 editor 必然喂进四个不同的条件前缀，
        是否真的产生不同的分布是**可检验的** ——
        scripts/diagnose_conditioning.py 就是 proposal 要求的那个 diagnostic。

预算口径（proposal §3.2 / §4.3）：每次调用正好采样 K 次，
非法/重复/破坏 scaffold 的提议照样算一次 generation attempt，不做过采样补偿。
所以 validity 是一个真实被测量的指标，而不是像 RDKit 池那样构造上恒为 100%。

确定性（interfaces.Editor 的硬要求）：采样用 torch.Generator，
种子由 (smiles, editor名, seed) 的 blake2b 导出 —— 不能用 Python hash()，
它每个进程带盐，会让 paired 比较失效。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

from .chem import canonical
from .generator import (
    CondSmilesTransformer, GenConfig, detokenize, make_prefix, tokenize,
)
from .interfaces import OBJECTIVES
from .oracles import OracleBank

# 解码参数：在 router 比较之前冻结（proposal §3.1）
TEMPERATURE = 0.9
TOP_K = 24

# 提议缓存：键是 (ckpt, editor名, 分子, k, seed, bump)，值是候选列表。
# 进程内有效 —— run_experiment 的评估阶段用 multiprocessing.Pool，
# 每个 worker 各有一份；训练阶段是单进程，repeat 之间命中率最高。
_PROP_CACHE: dict = {}
_CACHE_MAX = 300_000


@lru_cache(maxsize=4)
def load_backbone(ckpt_path: str):
    """按路径缓存：四个 editor 共享同一份权重，只加载一次。"""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GenConfig(**blob["cfg"])
    model = CondSmilesTransformer(cfg)
    model.load_state_dict(blob["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blob["vocab"], blob["gen_meta"]


def _stable_seed(*parts) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**31 - 1)


@dataclass
class GenEditor:
    """name 必须是 OBJECTIVES 之一：它决定把哪一维的目标档位往上推。"""

    name: str
    ckpt: str
    bank: OracleBank
    # 目标档位上推几格。1 太弱：gsk3b/jnk3 相邻档位的真实属性差只有 0.014
    # 量级（见 docs/findings.md §10），条件几乎没在要求任何东西。
    # bump >= n_bins 等价于"顶到最高档"，把四个条件的差异拉到最大。
    bump: int = 1
    _m: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._m = load_backbone(self.ckpt)
        _, _, meta = self._m
        self.obj_idx = list(meta["objectives"]).index(self.name)
        self.n_bins = meta["n_bins"]
        # "delta"：档位 = dy 的变化分位（scripts/rebin_delta.py）
        # "absolute"：档位 = 属性的绝对分位（旧口径，仅用于对照已归档的权重）
        self.mode = meta.get("conditioning", "absolute")
        if self.mode == "delta":
            self.neutral = meta["neutral_bin"]
        else:
            self.edges = meta["bin_edges"]

    def _abs_bins(self, smiles: str) -> list[int] | None:
        """当前分子的属性绝对档位（仅 absolute 口径用）。走 bank 的 cache ——
        env 已经给它打过分了，所以这里是 cache hit，不额外消耗 oracle 预算。"""
        v = self.bank.vector(smiles)
        if v is None:
            return None
        out = []
        for j, o in enumerate(OBJECTIVES):
            # 必须与训练时的分箱**逐字一致**：
            # scripts/prepare_generator_data.py:123 用
            # np.searchsorted(edges, y, side="right")，即 y >= edge 计一档。
            # 之前这里写的是 v[j] > x，只在"正好落在边界上"时不同 ——
            # 但 gsk3b/jnk3 的 TDC 分数高度离散（jnk3 第一条边界就是 0.0），
            # 实测 70.7% 的分子至少有一维因此拿到训练中没出现过的档位 token
            # （jnk3 63%、gsk3b 27%，qed/sa ~0%）。
            out.append(int(sum(1 for x in self.edges[o] if v[j] >= x)))
        return out

    def _target(self, smiles: str) -> list[int] | None:
        """这个 editor 要求的条件向量：只把自己那一维往上推，其余保持不变。"""
        if self.mode == "delta":
            # "保持不变" = dy 落在中间档；不依赖当前分子，所以不存在顶格问题
            tgt = [self.neutral[o] for o in OBJECTIVES]
        else:
            tgt = self._abs_bins(smiles)
            if tgt is None:
                return None
        tgt[self.obj_idx] = min(tgt[self.obj_idx] + self.bump, self.n_bins - 1)
        return tgt

    def propose(self, smiles: str, k: int, seed: int) -> list[str]:
        model, vocab, meta = self._m
        tgt = self._target(smiles)
        if tgt is None:
            return []

        # 条件 = 当前分子本身 + 推高一格的目标档位
        parent = canonical(smiles) or smiles
        prefix = make_prefix(tgt, tokenize(parent, vocab), vocab)
        if len(prefix) >= model.cfg.max_len:
            return []

        # 跨 policy / 跨 repeat 的提议缓存。env.py:167 明确要求所有 policy 在
        # 同一 (分子, expert, seed) 上拿到相同候选，所以这里本来就是纯函数；
        # 不缓存的话 6 个 policy 和 3 次 repeat 会把同样的采样重做一遍
        # （OracleBank 早就缓存了分数，提议之前没缓存）。
        ck = (self.ckpt, self.name, parent, k, seed, self.bump)
        hit = _PROP_CACHE.get(ck)
        if hit is not None:
            return list(hit)

        inv_eos = vocab["<eos>"]
        inv = {i: t for t, i in vocab.items()}
        out: list[str] = []
        seen = {parent}
        g = torch.Generator()
        g.manual_seed(_stable_seed(parent, self.name, seed))
        # 正好 K 次尝试，不补偿；合成一个 batch 采（见 generator.sample_batch）
        for ids in model.sample_batch(prefix, k, max_new=meta["max_len"] + 4,
                                      eos=inv_eos, temperature=TEMPERATURE,
                                      top_k=TOP_K, generator=g):
            cs = canonical(detokenize(ids, inv))   # 非法 -> None，照样耗一次尝试
            if cs is None or cs in seen:
                continue
            seen.add(cs)
            out.append(cs)   # scaffold 检查交给 env，越界提议要进统计

        if len(_PROP_CACHE) < _CACHE_MAX:
            _PROP_CACHE[ck] = list(out)
        return out


def make_gen_editors(ckpt: str, bank: OracleBank,
                     bump: int = 1) -> dict[str, GenEditor]:
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"找不到生成器 {ckpt}；先跑 scripts/train_generator.py")
    return {n: GenEditor(name=n, ckpt=ckpt, bank=bank, bump=bump)
            for n in OBJECTIVES}
