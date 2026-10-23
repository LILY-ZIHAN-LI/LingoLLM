"""生成 scaffold-split 的 seed 分子：200 train / 50 val / 100 test。

种子选择的两条原则（第一版诊断暴露的问题）：
1. **要有活性抓手。** 随机 ChEMBL 分子的 GSK3β/JNK3 预测分基本是 0，
   从零活性出发做取代基修饰造不出活性，HV 实际只由 QED/SA 驱动，
   四目标退化成两目标。所以按 max(gsk3b, jnk3) 排序优先选有信号的分子。
2. **要有提升空间。** ChEMBL 分子本来就在 QED 最优区，任何修饰都只会让它下降。
   卡 QED 上限，保证 router 有事可做。

用法: python scripts/make_seeds.py --out data/ --scan 40000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from src.chem import canonical, murcko_scaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_CANDIDATE_DATASETS = ["ChEMBL_V29", "ChEMBL", "ZINC", "MOSES"]

_BANK = None


def _init_worker():
    global _BANK
    from src.oracles import OracleBank
    _BANK = OracleBank()


def _score_one(smi: str):
    v = _BANK.vector(smi)
    return (smi, v)


def load_pool(n_raw: int) -> tuple[list[str], str]:
    from tdc.generation import MolGen

    last_err = None
    for name in _CANDIDATE_DATASETS:
        try:
            print(f"尝试下载 {name} ...")
            df = MolGen(name=name).get_data()
            col = "smiles" if "smiles" in df.columns else df.columns[0]
            return df[col].astype(str).tolist()[:n_raw], name
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  {name} 不可用: {type(e).__name__}")
    raise RuntimeError(f"没有可用数据集，最后错误: {last_err}")


def lead_like(smi: str) -> bool:
    """便宜的 RDKit 预筛，先把 scan 规模压下来再上 oracle。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    if not (250 <= Descriptors.MolWt(mol) <= 480):
        return False
    if mol.GetNumHeavyAtoms() > 42:
        return False
    # 必须有芳香 CH，否则增长型 editor 无从下手
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[cH]")):
        return False
    # 必须有可删的环上单原子取代，否则缩小型 editor 产不出候选
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R][!R;D1]")):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--scan", type=int, default=40000, help="送进 oracle 打分的分子数")
    ap.add_argument("--raw", type=int, default=400000, help="从数据集读取的原始条数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--max-qed", type=float, default=0.70,
                    help="QED 上限；已经很类药的分子没有优化余量")
    ap.add_argument("--min-act", type=float, default=0.15,
                    help="活性下限：低于此值 oracle 基本是平的，编辑没有梯度可循")
    ap.add_argument("--max-act", type=float, default=0.55,
                    help="活性上限：高于此值已接近 oracle 饱和，任何修饰都只会变差")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    need = args.n_train + args.n_val + args.n_test

    raw, source = load_pool(args.raw)
    print(f"数据源: {source}，原始 {len(raw)} 条")

    # --- 便宜预筛 ---
    pre: list[str] = []
    seen: set[str] = set()
    for smi in raw:
        c = canonical(smi)
        if c is None or c in seen or not lead_like(c):
            continue
        seen.add(c)
        pre.append(c)
        if len(pre) >= args.scan:
            break
    print(f"预筛后 {len(pre)} 个分子，送 oracle 打分（{args.workers} 进程）...")

    # --- 并行打分 ---
    with Pool(args.workers, initializer=_init_worker) as pool:
        scored = pool.map(_score_one, pre, chunksize=200)

    # vector 顺序: gsk3b, jnk3, qed, sa
    rows = [(s, v) for s, v in scored if v is not None]
    print(f"打分完成 {len(rows)} 条")

    act = [max(v[0], v[1]) for _, v in rows]
    print(f"  活性信号: max(gsk3b,jnk3) 中位数 {sorted(act)[len(act)//2]:.3f}，"
          f">0.1 的有 {sum(a > 0.1 for a in act)} 个，"
          f">0.3 的有 {sum(a > 0.3 for a in act)} 个")

    # --- 选种：卡在「有信号但未饱和」的活性带内，而不是取最活跃的 ---
    # 取最活跃的分子会让所有 editor 的活性 Δ 都为负 —— 没有提升空间，
    # 和 QED 已经最优时的问题完全对称。
    cands = [(s, v) for s, v in rows
             if v[2] <= args.max_qed
             and args.min_act <= max(v[0], v[1]) <= args.max_act]
    random.Random(args.seed).shuffle(cands)
    print(f"  QED<={args.max_qed} 且活性在 [{args.min_act}, {args.max_act}] 的有 "
          f"{len(cands)} 个")

    picked: list[tuple[str, list[float]]] = []
    used_scaffolds: set[str] = set()
    for s, v in cands:
        scaf = murcko_scaffold(s)
        if not scaf or scaf in used_scaffolds:
            continue
        used_scaffolds.add(scaf)
        picked.append((s, v))
        if len(picked) >= need:
            break

    if len(picked) < need:
        print(f"警告: 只选到 {len(picked)}/{need}。加大 --scan 或放宽 --max-qed")

    # --- scaffold 不重叠地切分 ---
    random.Random(args.seed).shuffle(picked)
    splits: dict[str, list[str]] = {}
    i = 0
    for name, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        splits[name] = [s for s, _ in picked[i:i + n]]
        i += n
        print(f"  {name}: {len(splits[name])}")

    sel_act = [max(v[0], v[1]) for _, v in picked]
    sel_qed = [v[2] for _, v in picked]
    print(f"\n选中种子: 活性中位数 {sorted(sel_act)[len(sel_act)//2]:.3f}，"
          f"QED 中位数 {sorted(sel_qed)[len(sel_qed)//2]:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, mols in splits.items():
        (out / f"seeds_{name}.txt").write_text("\n".join(mols) + "\n")
    (out / "seeds_meta.json").write_text(json.dumps({
        "source": source,
        "split_seed": args.seed,
        "n_scanned": len(pre),
        "n_scored": len(rows),
        "max_qed": args.max_qed,
        "selection": "QED<=max_qed, 按 max(gsk3b,jnk3) 降序, 每 scaffold 一个",
        "sizes": {k: len(v) for k, v in splits.items()},
        "scaffold_overlap": 0,
        "empty_scaffold_rule": "dropped",
        "median_activity": sorted(sel_act)[len(sel_act) // 2] if sel_act else None,
        "median_qed": sorted(sel_qed)[len(sel_qed) // 2] if sel_qed else None,
    }, indent=2))
    print(f"写入 {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())