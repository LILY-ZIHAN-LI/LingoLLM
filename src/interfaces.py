"""三人并行开发的接口契约。改这个文件前先在群里说一声。

依赖方向：editors.py / env.py / policies.py 都只 import 这里，互不 import。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

OBJECTIVES = ("gsk3b", "jnk3", "qed", "sa")
N_OBJ = len(OBJECTIVES)

# 四个 expert 的名字，等于 action space
EXPERTS = ("gsk3b", "jnk3", "qed", "sa")
N_ACTIONS = len(EXPERTS)


@dataclass(frozen=True)
class Candidate:
    """一次 proposal 的结果。非法/重复/破坏 scaffold 的也要返回，用于统计。"""

    smiles: str | None          # canonical SMILES；非法时为 None
    parent: str                 # 产生它的分子
    expert: str                 # 哪个 expert 提的
    valid: bool
    scaffold_ok: bool
    scores: tuple[float, ...] | None = None   # 按 OBJECTIVES 顺序，越大越好

    @property
    def eligible(self) -> bool:
        return self.valid and self.scaffold_ok


@dataclass
class StepLog:
    """每个 step 记一条，失败分析和 TA 要的诊断全靠它。"""

    t: int
    expert: str
    current: str
    n_proposed: int
    n_valid: int
    n_scaffold_ok: int
    n_novel: int                # 本 episode 内没见过的
    hv_before: float
    hv_after: float
    reward: float
    archive_size: int
    entered_archive: bool       # 诊断稀疏 reward 用：有没有候选进 archive
    best_single_gain: float     # 单目标最大提升，即使 HV 没涨也能看出 editor 有没有用


class Editor(Protocol):
    """所有 expert 的统一接口。实现见 editors.py。

    必须是确定性的：同样的 (smiles, seed) 必须给同样的输出，
    否则 proposal cache 失效，policy 之间就不是 paired 比较了。
    """

    name: str

    def propose(self, smiles: str, k: int, seed: int) -> list[str]:
        """返回至多 k 个候选 SMILES 字符串（可能非法，由 env 负责校验）。"""
        ...


class Policy(Protocol):
    """所有 router 的统一接口。实现见 policies.py（Yefeng）。"""

    name: str

    def act(self, obs: Sequence[float], t: int) -> int:
        """返回 action index，对应 EXPERTS[i]。"""
        ...

    def update(self, *args, **kwargs) -> None:
        """无学习的 baseline 留空实现即可。"""
        ...


@dataclass
class EpisodeConfig:
    """所有 policy 必须共用同一份，否则不是 matched budget。"""

    max_steps: int = 20
    k_per_call: int = 4
    enforce_scaffold: bool = True
    ref_point: tuple[float, ...] = field(default=(0.0,) * N_OBJ)
    lam: float = 0.0            # call penalty 系数
    cost_unit: str = "generation"  # "generation" | "oracle"；见 README 的说明
