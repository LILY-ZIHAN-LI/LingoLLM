"""go/no-go 诊断：四个 editor 是否真的各有专长？

这是训 DQN 之前**必须**跑的一步。如果每个 editor 在它自己的目标上
并不比其他 editor 更强，那么 routing 在原理上就不可能有收益，
后面无论 DQN 还是 bandit 都只能得到 null result。
提前知道可以改编辑池，而不是在 report 里解释为什么没效果。

用法: python scripts/check_editor_specialization.py --seeds data/seeds_train.txt -n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import scaffold_preserved  # noqa: E402
from src.editors import make_editors  # noqa: E402
from src.interfaces import OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/seeds_train.txt")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    mols = Path(args.seeds).read_text().split()[: args.n]
    editors = make_editors()
    bank = OracleBank()

    # 测三个量，其中 best_delta 才是 environment 实际利用的：
    #   mean_delta  - 所有候选的平均提升（被坏编辑稀释，参考价值有限）
    #   best_delta  - 每次调用里最好候选的提升（transition rule 取的就是这个）
    #   p_positive  - 该次调用至少产出一个正提升候选的比例
    mean_d = {e: {o: [] for o in OBJECTIVES} for e in editors}
    best_d = {e: {o: [] for o in OBJECTIVES} for e in editors}
    stats = {e: {"proposed": 0, "valid": 0, "scaf_ok": 0} for e in editors}

    for idx, parent in enumerate(mols):
        base = bank.vector(parent)
        if base is None:
            continue
        for ename, ed in editors.items():
            cands = ed.propose(parent, args.k, seed=idx)
            stats[ename]["proposed"] += args.k
            per_call = {o: [] for o in OBJECTIVES}
            for c in cands:
                v = bank.vector(c)
                if v is None:
                    continue
                stats[ename]["valid"] += 1
                if not scaffold_preserved(parent, c):
                    continue
                stats[ename]["scaf_ok"] += 1
                for j, o in enumerate(OBJECTIVES):
                    d = v[j] - base[j]
                    mean_d[ename][o].append(d)
                    per_call[o].append(d)
            for o in OBJECTIVES:
                if per_call[o]:
                    best_d[ename][o].append(max(per_call[o]))

    def _show(table, title):
        print(f"\n{title}")
        print(f"{'':8}" + "".join(f"{o:>12}" for o in OBJECTIVES))
        mat = []
        for ename in editors:
            row = [float(np.mean(table[ename][o])) if table[ename][o] else 0.0
                   for o in OBJECTIVES]
            mat.append(row)
            print(f"{ename:8}" + "".join(f"{v:>12.4f}" for v in row))
        return np.array(mat)

    _show(mean_d, "平均 Δ（所有候选）")
    mat = _show(best_d, "最佳候选 Δ（environment 实际利用的量）")

    # 正确的判据是按列看：对每个目标，是否存在一个 editor 明显比别人强。
    names = list(editors)
    print("\n按目标看谁最强（列最大）：")
    col_wins = 0
    for j, o in enumerate(OBJECTIVES):
        col = mat[:, j]
        order = np.argsort(-col)
        top, second = names[order[0]], col[order[0]]
        gap = col[order[0]] - col[order[1]]
        hit = "✓" if top == o else " "
        print(f"  {o:6} -> {top:8} ({second:+.4f}, 领先第二名 {gap:.4f}) {hit}")
        if top == o:
            col_wins += 1

    print("\n每次调用至少产出一个正提升的比例：")
    for ename in editors:
        row = " ".join(
            f"{o}={np.mean([d > 0 for d in best_d[ename][o]]):.0%}" if best_d[ename][o]
            else f"{o}=n/a" for o in OBJECTIVES)
        print(f"  {ename:8} {row}")

    print("\n有效率：")
    for ename, s in stats.items():
        p = s["proposed"] or 1
        print(f"  {ename:8} valid {s['valid']/p:5.1%}  scaffold_ok {s['scaf_ok']/p:5.1%}")

    diag_wins = col_wins
    print(f"\n各目标由对应 editor 领先的数量: {diag_wins}/4")
    print(f"\noracle: {bank.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())