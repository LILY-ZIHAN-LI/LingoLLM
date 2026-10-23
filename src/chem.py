"""分子合法性与 scaffold 约束检查。proposal 里 "validity/scaffold checks" 这部分。"""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=200_000)
def canonical(smiles: str) -> str | None:
    """非法返回 None。带缓存，全项目统一走这个函数。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=200_000)
def murcko_scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold 的 canonical SMILES。

    无环分子的 scaffold 是空字符串 "" —— proposal 里说要"显式记录的分组规则"，
    这里的规则是：空 scaffold 单独成一组，不与任何有环分子混在一起。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        # 去立体化学：否则 SA editor 的 strip_stereo 会被误判成破坏 scaffold。
        # scaffold 约束管的是骨架连接性，不是构型。
        Chem.RemoveStereochemistry(scaf)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def scaffold_preserved(parent: str, child: str) -> bool:
    """child 是否保留了 parent 的 Murcko scaffold。"""
    ps, cs = murcko_scaffold(parent), murcko_scaffold(child)
    if ps is None or cs is None:
        return False
    return ps == cs


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return 0 if mol is None else mol.GetNumHeavyAtoms()


def is_reasonable(smiles: str, max_heavy: int = 70) -> bool:
    """挡掉 editor 偶尔产生的退化分子（空分子、超大聚合物）。

    注意：这不是化学合理性判断，只是防止 oracle 被喂垃圾拖慢。
    被挡下的仍然计入 generation attempt。
    """
    n = heavy_atom_count(smiles)
    return 3 <= n <= max_heavy
