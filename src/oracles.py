"""四目标 oracle 封装：统一方向（越大越好）、统一缓存、统一计数。

所有 policy 共用同一个实例，这样 c_t（新增 scalar 打分次数）的口径对所有
baseline 完全一致 —— 这是 reward 里 -λ·c_t 项可信的前提。
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

# --- PyTDC 1.1.x 残留的 `from rdkit.six import iteritems`，先补一个 shim ---
if "rdkit.six" not in sys.modules:
    _six = types.ModuleType("rdkit.six")
    _six.iteritems = lambda d: iter(d.items())
    sys.modules["rdkit.six"] = _six

from rdkit import Chem, RDLogger  # noqa: E402
from tdc import Oracle  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OBJECTIVES = ("gsk3b", "jnk3", "qed", "sa")

# SA 原始范围 1-10，越小越好 -> 映射到 [0,1] 越大越好
_SA_LO, _SA_HI = 1.0, 10.0


def canonical(smiles: str) -> str | None:
    """返回 canonical SMILES；非法分子返回 None。"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@dataclass
class OracleBank:
    """四个目标的打分器。score() 返回 dict，方向统一为「越大越好」。"""

    _oracles: dict = field(default_factory=dict, init=False)
    _cache: dict = field(default_factory=dict, init=False)
    n_calls: int = field(default=0, init=False)  # 实际计算的 (分子, 目标) 对数
    n_hits: int = field(default=0, init=False)   # 缓存命中数

    def __post_init__(self) -> None:
        self._oracles = {
            "gsk3b": Oracle(name="GSK3B"),
            "jnk3": Oracle(name="JNK3"),
            "qed": Oracle(name="QED"),
            "sa": Oracle(name="SA"),
        }

    def score(self, smiles: str) -> dict[str, float] | None:
        """非法 SMILES 返回 None（调用方应记为一次失败的 proposal）。"""
        key = canonical(smiles)
        if key is None:
            return None
        if key in self._cache:
            self._hits_add(len(OBJECTIVES))
            return self._cache[key]

        raw = {name: float(orc(key)) for name, orc in self._oracles.items()}
        self.n_calls += len(OBJECTIVES)
        vec = {
            "gsk3b": raw["gsk3b"],
            "jnk3": raw["jnk3"],
            "qed": raw["qed"],
            "sa": (_SA_HI - raw["sa"]) / (_SA_HI - _SA_LO),
        }
        vec = {k: min(max(v, 0.0), 1.0) for k, v in vec.items()}
        self._cache[key] = vec
        return vec

    def _hits_add(self, n: int) -> None:
        self.n_hits += n

    def vector(self, smiles: str) -> list[float] | None:
        v = self.score(smiles)
        return None if v is None else [v[k] for k in OBJECTIVES]

    def stats(self) -> dict[str, int]:
        return {"logical_calls": self.n_calls, "cache_hits": self.n_hits,
                "unique_molecules": len(self._cache)}
