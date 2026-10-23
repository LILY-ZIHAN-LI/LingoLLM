"""action space 是不是退化的：四个 editor 实际上有几个不同的"方向"。

背景：check_editor_specialization.py 报出"4 个目标里只有 2 个由对应 editor 领先"，
但那是点估计，没有不确定性。这个脚本做三件事：

  1. 对每个目标 j，检验"editor j 是否真的比最强的其他 editor 更好"，
     用按种子配对的 bootstrap 给置信区间。
  2. 算 editor 之间改进向量的余弦相似度：相似度接近 1 的两个 editor
     对 router 来说是可互换的，路由在它们之间做选择不产生任何差别。
  3. 用 participation ratio 报 action space 的有效维度：
     名义 4 个 action，实际有几个不同的方向。

这条是整个项目的前提条件：如果 action space 本身是退化的，
那"学出来的路由没有显著优势"就不是关于路由的结论，而是关于编辑池的结论。
两者必须在 report 里分开讲。

  python scripts/analyze_action_space.py --n-seeds 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    objective_scales, participation_ratio,
)
from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.editors import make_editors  # noqa: E402
from src.gen_editors import make_gen_editors  # noqa: E402
from src.interfaces import EXPERTS, OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}


def best_delta_per_seed(seeds, editors, bank, k: int):
    """D[s, e, j] = 第 s 个种子上 editor e 的候选里，目标 j 的最大提升。

    用"最佳候选"而不是"平均候选"，因为 env 的 transition rule 取的就是最佳。
    该次调用没有任何合格候选时记 NaN（而不是 0）：
    记 0 会把"提不出东西"伪装成"提了但没提升"。
    """
    n_e, n_o = len(EXPERTS), len(OBJECTIVES)
    D = np.full((len(seeds), n_e, n_o), np.nan)
    for s, smi in enumerate(seeds):
        parent = canonical(smi)
        if parent is None:
            continue
        base = bank.vector(parent)
        if base is None:
            continue
        base = np.asarray(base)
        for ei, e in enumerate(EXPERTS):
            best = None
            for cand in editors[e].propose(parent, k, seed=s):
                c = canonical(cand)
                if c is None or not scaffold_preserved(parent, c):
                    continue
                v = bank.vector(c)
                if v is None:
                    continue
                d = np.asarray(v) - base
                best = d if best is None else np.maximum(best, d)
            if best is not None:
                D[s, ei] = best
        if (s + 1) % 20 == 0:
            print(f"  {s+1}/{len(seeds)}")
    return D


def boot_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = np.nanmean(x[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_test.txt")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="results/action_space.json")
    ap.add_argument("--pairs", default="data/gen_pairs_delta.jsonl",
                    help="取各目标可达变化的尺度，用于归一 eff dim")
    ap.add_argument("--editors", choices=["rdkit", "gen"], default="rdkit",
                    help="rdkit = 确定性反应池（proposal 的 control）；"
                         "gen = 条件生成器（proposal §3.1 的主实验）")
    ap.add_argument("--ckpt", default="results_gen/generator.pt")
    ap.add_argument("--pool-overlap", type=float, default=0.0,
                    help="仅 rdkit：池重叠比例，见 src/editors.py:_mix")
    ap.add_argument("--qed-disjoint", action="store_true",
                    help="仅 rdkit：qed 偏重时用 _QED_SWAP 而非 _SHRINK")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]
    bank = OracleBank()
    editors = (make_gen_editors(args.ckpt, bank) if args.editors == "gen"
               else make_editors(overlap=args.pool_overlap,
                                 qed_disjoint=args.qed_disjoint))
    print(f"editor 池: {args.editors}"
          + (f"（{args.ckpt}）" if args.editors == "gen" else ""))

    print(f"在 {len(seeds)} 个种子上取各 editor 的最佳候选提升...")
    D = best_delta_per_seed(seeds, editors, bank, args.k)

    cover = (~np.isnan(D[:, :, 0])).mean(axis=0)
    print("\n各 editor 提出至少一个合格候选的比例：")
    print("  " + "  ".join(f"{LABEL[e]}:{c:.0%}" for e, c in zip(EXPERTS, cover)))

    M = np.nanmean(D, axis=0)          # 4 editors x 4 objectives
    print("\n平均最佳提升矩阵（行=editor，列=目标）")
    print(f"  {'editor':<8}" + "".join(f"{LABEL[o]:>10}" for o in OBJECTIVES))
    for ei, e in enumerate(EXPERTS):
        print(f"  {LABEL[e]:<8}" + "".join(f"{M[ei, j]:>10.4f}"
                                           for j in range(len(OBJECTIVES))))

    # ---------- 1. 每个目标：对应 editor 真的最强吗 ----------
    print("\n配对检验：editor j 在目标 j 上是否强过最好的其他 editor")
    print("  （按种子配对，bootstrap 95% CI；两边都有合格候选的种子才计入）")
    spec = {}
    for j, o in enumerate(OBJECTIVES):
        own = D[:, j, j]
        others = np.delete(D[:, :, j], j, axis=1)
        rival_idx = int(np.nanargmax(np.nanmean(others, axis=0)))
        rival_name = [e for i, e in enumerate(EXPERTS) if i != j][rival_idx]
        rival = others[:, rival_idx]
        ok = ~np.isnan(own) & ~np.isnan(rival)
        d = own[ok] - rival[ok]
        lo, hi = boot_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        win = "✓" if lo > 0 else ("✗" if hi < 0 else "~")
        spec[o] = {"rival": rival_name, "mean": float(d.mean()),
                   "lo": lo, "hi": hi, "n": int(ok.sum()),
                   "specialized": bool(lo > 0)}
        print(f"  {LABEL[o]:<6} vs {LABEL[rival_name]:<6}"
              f"{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}  n={ok.sum():<4}{win}")
    n_spec = sum(v["specialized"] for v in spec.values())
    print(f"\n  真正专精自己目标的 editor：{n_spec}/4"
          "（✓ = 显著强于最强对手，~ = 无显著差异，✗ = 显著更弱）")

    # ---------- 2. editor 两两相似度 ----------
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    C = Mn @ Mn.T
    print("\neditor 改进向量的余弦相似度（接近 1 = 对 router 可互换）")
    print("  " + " " * 8 + "".join(f"{LABEL[e]:>9}" for e in EXPERTS))
    for ei, e in enumerate(EXPERTS):
        print(f"  {LABEL[e]:<8}" + "".join(f"{C[ei, j]:>9.3f}"
                                           for j in range(len(EXPERTS))))
    pairs = [(EXPERTS[i], EXPERTS[j], float(C[i, j]))
             for i in range(len(EXPERTS)) for j in range(i + 1, len(EXPERTS))]
    for a, b, v in sorted(pairs, key=lambda t: -t[2])[:2]:
        print(f"  最相似的一对：{LABEL[a]} - {LABEL[b]} = {v:.3f}")

    # ---------- 3. action space 的有效维度 ----------
    sv = np.linalg.svd(M, compute_uv=False)
    eff = participation_ratio(M)
    # 按各目标的可达变化尺度归一后再算：不归一时 dy 尺度最大的目标会独占
    # 奇异值（见 analysis.objective_scales 与 docs/findings.md §10d）。
    s = objective_scales(args.pairs)
    eff_n = participation_ratio(M / s) if s is not None else float("nan")
    print(f"\n  奇异值 {np.round(sv, 4).tolist()}")
    print(f"  有效维度（未归一）     = {eff:.2f} / 4.00")
    print(f"  有效维度（按目标归一） = {eff_n:.2f} / 4.00"
          f"   可达上限 3.45")
    print(f"  -> 名义 4 个 editor，实际约 {eff_n:.1f} 个不同的编辑方向。")

    Path(args.out).write_text(json.dumps({
        "n_seeds": len(seeds), "coverage": cover.round(4).tolist(),
        "mean_best_delta": M.round(5).tolist(),
        "specialization": spec,
        "editor_cosine": C.round(4).tolist(),
        "singular_values": sv.round(5).tolist(),
        "effective_action_dim": eff,
        "effective_action_dim_scaled": eff_n,
        "achievable_eff_dim_scaled": 3.45,
        "n_specialized": int(n_spec),
    }, indent=2))
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
