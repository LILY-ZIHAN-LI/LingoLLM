"""生成器训练语料：从 ChEMBL 里切出 (scaffold, 属性目标) -> SMILES 的重建数据。

对应 proposal §3.1 的 fallback 方案：
  "train a compact SMILES transformer ... learn to recover molecular SMILES from
   scaffold SMILES and discretized property targets using token cross-entropy"

泄漏控制（proposal §4.1）：
  - 按 Bemis-Murcko scaffold 划分，**任何与 val/test 种子同 scaffold 的分子一律剔除**。
    router 的 held-out 种子不能以任何形式出现在生成器的训练集里。
  - 训练种子自己的 scaffold 保留（router 本来就在这些 scaffold 上训练）。
  - empty scaffold 的分子丢弃，和 seeds_meta.json 的规则一致。

属性分箱：四个目标各切 n_bins 档，档位边界**只在训练集上**拟合，写进 meta。
推理时把某一个目标的档位往上推一格、其余保持 -> 这就是 objective-conditioned editor。

  python scripts/prepare_generator_data.py --n-target 100000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RDLogger.DisableLog("rdApp.*")

from src.chem import canonical, murcko_scaffold  # noqa: E402
from src.interfaces import OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

# 太长的 SMILES 会把 batch 撑爆，而且多半是肽/天然产物，不是 lead optimization 的对象
MAX_LEN = 110
MIN_LEN = 12


def held_out_scaffolds(data: Path) -> set[str]:
    """val + test 种子的 scaffold，连同它们自己，全部不得进入生成器训练集。"""
    out: set[str] = set()
    for name in ("seeds_val.txt", "seeds_test.txt"):
        for smi in data.joinpath(name).read_text().split():
            sc = murcko_scaffold(smi)
            if sc:
                out.add(sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl", default="data/chembl_v29.csv")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/gen_corpus.jsonl")
    ap.add_argument("--meta", default="data/gen_meta.json")
    ap.add_argument("--n-target", type=int, default=100000,
                    help="目标语料规模；proposal 写的是 50k-100k")
    ap.add_argument("--n-scan", type=int, default=600000,
                    help="最多扫多少行 ChEMBL")
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    data = Path(args.data)
    banned = held_out_scaffolds(data)
    print(f"held-out scaffold（val+test 种子）: {len(banned)} 个，将被剔除")

    bank = OracleBank()
    rows: list[dict] = []
    seen: set[str] = set()
    n_scanned = n_badlen = n_leak = n_noscaf = 0

    with open(args.chembl) as fh:
        rd = csv.DictReader(fh)
        for rec in rd:
            n_scanned += 1
            if n_scanned > args.n_scan or len(rows) >= args.n_target:
                break
            smi = rec.get("smiles") or ""
            if not (MIN_LEN <= len(smi) <= MAX_LEN):
                n_badlen += 1
                continue
            cs = canonical(smi)
            if cs is None or cs in seen or len(cs) > MAX_LEN:
                continue
            mol = Chem.MolFromSmiles(cs)
            if mol is None:
                continue
            sc = murcko_scaffold(cs)
            if not sc:
                n_noscaf += 1
                continue
            if sc in banned:          # 泄漏控制：held-out scaffold 一律不要
                n_leak += 1
                continue
            seen.add(cs)
            rows.append({"smiles": cs, "scaffold": sc,
                         "y": [float(v) for v in bank.vector(cs)]})
            if len(rows) % 10000 == 0:
                print(f"  收集 {len(rows)} / 已扫 {n_scanned}")

    print(f"\n扫描 {n_scanned} 行 -> 保留 {len(rows)}")
    print(f"  长度不合格 {n_badlen} / 无 scaffold {n_noscaf} / "
          f"撞上 held-out scaffold {n_leak}")

    # --- 属性分箱：边界只在训练片段上拟合 -------------------------------
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val_frac)
    val_idx, tr_idx = set(idx[:n_val].tolist()), idx[n_val:]

    Y = np.array([rows[i]["y"] for i in tr_idx])
    qs = np.linspace(0, 100, args.n_bins + 1)[1:-1]
    edges = {o: np.percentile(Y[:, j], qs).tolist()
             for j, o in enumerate(OBJECTIVES)}
    print("\n属性分箱边界（仅用训练片段拟合）:")
    for o in OBJECTIVES:
        print(f"  {o:6} {[round(v, 4) for v in edges[o]]}")

    def to_bins(y):
        return [int(np.searchsorted(edges[o], y[j], side="right"))
                for j, o in enumerate(OBJECTIVES)]

    out = Path(args.out)
    with out.open("w") as fh:
        for i, r in enumerate(rows):
            r["bins"] = to_bins(r["y"])
            r["split"] = "val" if i in val_idx else "train"
            fh.write(json.dumps(r) + "\n")

    meta = {
        "n_total": len(rows), "n_val": n_val, "n_bins": args.n_bins,
        "bin_edges": edges, "objectives": list(OBJECTIVES),
        "max_len": MAX_LEN, "n_scanned": n_scanned,
        "n_dropped_heldout_scaffold": n_leak,
        "banned_scaffolds": len(banned),
        "leakage_rule": "val/test 种子的 scaffold 及其同 scaffold 分子全部剔除",
        "oracle_calls": bank.n_calls,
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    print(f"\n写入 {out}（{len(rows)} 条，val {n_val}）和 {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
