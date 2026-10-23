"""四个 objective-conditioned editor（RDKit 确定性编辑池）。

设计要点（第一版诊断暴露的问题）：
1. QED 是 MW/logP/TPSA 的钟形函数，固定反应池服务不了它 —— 分子偏大时该删、
   偏小时该加。所以 qed/sa editor 按 RDKit 描述符切换子池。描述符是免费计算的
   结构特征，不是目标本身，不计入 oracle 预算。
2. SA editor 需要对任意分子都能触发的通用删除反应，否则大部分种子上它产不出候选。
3. gsk3b/jnk3 两个池在化学上刻意不重叠。但两个 RF oracle 本身高度相关（都是激酶），
   它们是否真能分化是实证问题，不是设计能保证的 —— 见 check_editor_specialization.py。
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors

from .chem import canonical, is_reasonable

RDLogger.DisableLog("rdApp.*")

# --- GSK3B：极性 / 氢键给体受体 ------------------------------------------
_GSK3B = [
    "[cH:1]>>[c:1]N",
    "[cH:1]>>[c:1]O",
    "[cH:1]>>[c:1]C(N)=O",
    "[cH:1]>>[c:1]NC(C)=O",
    "[cH:1]>>[c:1]C#N",
    "[cH:1]>>[c:1]S(N)(=O)=O",
    "[cH:1]>>[c:1]N1CCNCC1",
    "[cH:1]>>[c:1]NC(N)=O",
    "[cH:1]>>[c:1]C(=O)O",
]

# --- JNK3：卤素 / 疏水 / 硫 ----------------------------------------------
_JNK3 = [
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]Cl",
    "[cH:1]>>[c:1]Br",
    "[cH:1]>>[c:1]C(F)(F)F",
    "[cH:1]>>[c:1]OC(F)(F)F",
    "[cH:1]>>[c:1]SC",
    "[cH:1]>>[c:1]C(C)C",
    "[cH:1]>>[c:1]C(C)(C)C",
    "[cH:1]>>[c:1]OCC",
]

# --- 通用删除：对任意带环分子都能触发（SA / QED-缩小 共用）----------------
_SHRINK = [
    "[R:1][!R;D1]>>[R:1]",                 # 砍掉环上任意单原子取代
    "[c:1][C;!R;H3]>>[c:1]",               # 去芳环甲基
    "[c:1][F,Cl,Br,I]>>[c:1]",             # 去卤
    "[c:1]C(F)(F)F>>[c:1]C",               # CF3 -> Me
    "[c:1]S(N)(=O)=O>>[c:1]",              # 去磺酰胺
    "[c:1][N+](=O)[O-]>>[c:1]",            # 去硝基
    "[c:1]C#N>>[c:1]C",
    "[c:1]OC(F)(F)F>>[c:1]OC",
    "[C;!R;H2:1][C;!R;H3]>>[C;!R;H3:1]",   # 缩短烷基链
    "[c:1][O;!R][CH3]>>[c:1]O",            # 去甲基化
]

# --- 通用增长：往极性小基团方向加（QED-增大 用）--------------------------
_GROW = [
    "[cH:1]>>[c:1]O",
    "[cH:1]>>[c:1]N",
    "[cH:1]>>[c:1]OC",
    "[cH:1]>>[c:1]C",
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]C(N)=O",
]

# --- QED：等价替换而非删除，和 _SHRINK 不共享任何反应 ---
_QED_SWAP = [
    "[c:1]C(C)(C)C>>[c:1]C",
    "[c:1]Br>>[c:1]F",
    "[c:1]I>>[c:1]Cl",
    "[c:1]C(F)(F)F>>[c:1]C",
    "[c:1][N+](=O)[O-]>>[c:1]N",
    "[c:1]OC(F)(F)F>>[c:1]OC",
    "[c:1]C(=O)OC>>[c:1]C(=O)O",
    "[c:1]S(C)(=O)=O>>[c:1]C",
]

def _compile(smarts_list):
    out = []
    for s in smarts_list:
        rxn = AllChem.ReactionFromSmarts(s)
        if rxn is not None:
            out.append(rxn)
    return out


# 全部反应的并集，供 _mix 往各自的池里掺。
_ALL_SMARTS = sorted(set(_GSK3B + _JNK3 + _SHRINK + _GROW + _QED_SWAP))


def _mix(own: list[str], alpha: float) -> list[str]:
    """按 alpha 把并集里的反应确定性地掺进 own。alpha=0 原样，alpha=1 全并。

    排序键用 blake2b 而不是列表书写顺序：取前缀得到的是一列**嵌套**的池子，
    所以 alpha 单调增 => 每个池单调变大、四个 editor 单调趋同，
    alpha=1 时四个 editor 拿到同一个并集，在构造上完全等价
    （有效 action 维度应该塌到 1）。这是把 findings §11c
    "两个点不成曲线" 变成一条曲线用的旋钮。
    """
    if alpha <= 0:
        return list(own)
    have = set(own)
    extra = [x for x in _ALL_SMARTS if x not in have]
    extra.sort(key=lambda x: hashlib.blake2b(x.encode(),
                                             digest_size=8).digest())
    return list(own) + extra[:int(round(alpha * len(extra)))]


def _too_heavy(mol) -> bool:
    """QED 的钟形中心大致在 MW 300-380 / logP 1-3。超出就该缩，不足就该长。"""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    return mw > 360 or logp > 3.0


@dataclass
class RDKitEditor:
    """确定性：(smiles, seed) 相同则输出相同。proposal cache 成立的前提。"""

    name: str
    strip_stereo: bool = False
    # 并池比例，见 _mix。0 = 池子按设计各自独立（历史行为）。
    overlap: float = 0.0
    # qed 偏重时改用 _QED_SWAP 而不是 _SHRINK。默认 False 是历史行为：
    # _QED_SWAP 一直定义并编译着，但 _pools 从没返回过它，所以 qed 和 sa
    # 在偏重的分子上**字面共用同一个池** —— 这就是 findings §1b 里
    # "qed/sa 在 70/100 个种子上提议完全相同" 的来源。打开它把维度往上推。
    qed_disjoint: bool = False
    _heavy: list = field(default=None, init=False, repr=False)
    _light: list = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name == "gsk3b":
            heavy = light = _GSK3B
        elif self.name == "jnk3":
            heavy = light = _JNK3
        elif self.name == "sa":
            heavy = light = _SHRINK
        else:   # qed：按描述符条件化（偏重则缩，偏轻则长）
            heavy = _QED_SWAP if self.qed_disjoint else _SHRINK
            light = _GROW
        # 编译一次放实例上。propose 会把产物 sort 后再抽样，
        # 所以池内顺序不影响输出，overlap=0 时与改动前逐字等价。
        self._heavy = _compile(_mix(heavy, self.overlap))
        self._light = _compile(_mix(light, self.overlap))

    def _pools(self, mol) -> list:
        if self.name != "qed":
            return self._heavy
        return self._heavy if _too_heavy(mol) else self._light

    def propose(self, smiles: str, k: int, seed: int) -> list[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        products: list[str] = []
        seen: set[str] = {canonical(smiles) or smiles}

        for rxn in self._pools(mol):
            try:
                outcomes = rxn.RunReactants((mol,))
            except Exception:
                continue
            for tup in outcomes:
                for p in tup:
                    try:
                        Chem.SanitizeMol(p)
                        smi = Chem.MolToSmiles(p)
                    except Exception:
                        continue
                    if smi in seen or not is_reasonable(smi):
                        continue
                    seen.add(smi)
                    products.append(smi)

        if self.strip_stereo:
            flat = Chem.MolFromSmiles(smiles)
            if flat is not None and Chem.FindMolChiralCenters(
                flat, includeUnassigned=False
            ):
                Chem.RemoveStereochemistry(flat)
                smi = Chem.MolToSmiles(flat)
                if smi not in seen:
                    seen.add(smi)
                    products.append(smi)

        if not products:
            return []

        products.sort()
        rng = random.Random(seed)
        if len(products) <= k:
            return products
        return rng.sample(products, k)


def make_editors(overlap: float = 0.0,
                 qed_disjoint: bool = False) -> dict[str, RDKitEditor]:
    """默认参数复现改动前的池子，所以 §1-§7 的结果仍然可复现。"""
    kw = {"overlap": overlap, "qed_disjoint": qed_disjoint}
    return {
        "gsk3b": RDKitEditor("gsk3b", **kw),
        "jnk3": RDKitEditor("jnk3", **kw),
        "qed": RDKitEditor("qed", **kw),
        "sa": RDKitEditor("sa", strip_stereo=True, **kw),
    }
