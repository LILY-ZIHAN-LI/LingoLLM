"""四个目标到底张成几维：Spearman 相关 + PCA 有效维度。

为什么这件事重要：
  助教的 feedback 担心"目标太少留不下路由空间"。我们用的是四目标
  （GSK3B / JNK3 / QED / SA），这是 MAMO、MultiMol、RationaleRL 共用的标准组合。
  但"四个目标"不等于"四个独立方向"——如果 GSK3β 和 JNK3 高度相关
  （两者都是 ECFP6 特征上的随机森林激酶分类器），QED 和 SA 高度相关
  （都偏好小而简单的分子），那实际的权衡空间只有 2 维，
  路由能腾挪的余地天然就小。这一条据我们所知没有人验证过。

做法：在测试种子及其一步编辑候选上取分数矩阵，算两两 Spearman，
再对标准化后的分数做 PCA，用 participation ratio 报"有效维度数"。

  python scripts/analyze_objectives.py --n-seeds 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.editors import make_editors  # noqa: E402
from src.interfaces import EXPERTS, OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}


def spearman(X: np.ndarray) -> np.ndarray:
    """秩相关：目标分数的分布很偏（JNK3 大量接近 0），Pearson 会被尾部主导。"""
    R = np.apply_along_axis(lambda c: np.argsort(np.argsort(c)), 0, X).astype(float)
    R -= R.mean(axis=0)
    denom = np.sqrt((R ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    return (R.T @ R) / np.outer(denom, denom)


def effective_dim(evals: np.ndarray) -> float:
    """participation ratio：(Σλ)² / Σλ²。

    4 个完全独立且等方差的目标给出 4.0；两两完全重合的两组给出 2.0。
    比"解释 90% 方差需要几个主成分"更连续，不依赖阈值。
    """
    return float(evals.sum() ** 2 / np.maximum((evals ** 2).sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_test.txt")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="results/objective_analysis.json")
    ap.add_argument("--npz", default="results/objective_scores.npz")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]
    bank = OracleBank()
    editors = make_editors()

    rows, origin = [], []
    print(f"在 {len(seeds)} 个种子 + 它们的一步编辑候选上取分数...")
    for i, smi in enumerate(seeds):
        c0 = canonical(smi)
        if c0 is None:
            continue
        v = bank.vector(c0)
        if v is not None:
            rows.append(v)
            origin.append("seed")
        for e in EXPERTS:
            for cand in editors[e].propose(c0, args.k, seed=i):
                c = canonical(cand)
                if c is None or not scaffold_preserved(c0, c):
                    continue
                v = bank.vector(c)
                if v is not None:
                    rows.append(v)
                    origin.append(e)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(seeds)}  累计 {len(rows)} 个分子")

    X = np.array(rows, dtype=float)
    origin = np.array(origin)
    print(f"\n分数矩阵 {X.shape}（种子 {int((origin=='seed').sum())} + "
          f"合格候选 {int((origin!='seed').sum())}）\n")

    # ---------- 每个目标的分布 ----------
    print("各目标的分布（方向已统一为越大越好）")
    print(f"  {'目标':<8}{'均值':>8}{'标准差':>8}{'中位':>8}{'p90':>8}{'>0.5占比':>10}")
    dist = {}
    for j, o in enumerate(OBJECTIVES):
        c = X[:, j]
        dist[o] = {"mean": float(c.mean()), "std": float(c.std()),
                   "median": float(np.median(c)), "p90": float(np.percentile(c, 90)),
                   "frac_above_half": float((c > 0.5).mean())}
        print(f"  {LABEL[o]:<8}{c.mean():>8.3f}{c.std():>8.3f}"
              f"{np.median(c):>8.3f}{np.percentile(c,90):>8.3f}"
              f"{(c>0.5).mean():>10.1%}")
    print("\n  GSK3B/JNK3 的均值和方差都很小 = 大部分分子在两个激酶上都接近 0，"
          "\n  archive 的 HV 主要由 QED/SA 这两维撑起来。")

    # ---------- 两两相关 ----------
    S = spearman(X)
    print("\n两两 Spearman 相关")
    print("  " + " " * 8 + "".join(f"{LABEL[o]:>9}" for o in OBJECTIVES))
    for i, o in enumerate(OBJECTIVES):
        print(f"  {LABEL[o]:<8}" + "".join(f"{S[i, j]:>9.3f}"
                                           for j in range(len(OBJECTIVES))))

    pairs = [(i, j) for i in range(len(OBJECTIVES))
             for j in range(i + 1, len(OBJECTIVES))]
    strong = [(OBJECTIVES[i], OBJECTIVES[j], S[i, j]) for i, j in pairs
              if abs(S[i, j]) > 0.3]
    if strong:
        print("\n  |ρ| > 0.3 的目标对：")
        for a, b, v in sorted(strong, key=lambda t: -abs(t[2])):
            print(f"    {LABEL[a]} - {LABEL[b]}: {v:+.3f}")

    # ---------- PCA ----------
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
    C = np.cov(Z, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    ratio = evals / evals.sum()

    print("\nPCA（在标准化分数上；相关矩阵的特征分解）")
    print(f"  {'PC':<5}{'特征值':>9}{'方差占比':>10}{'累计':>9}   载荷")
    for i in range(len(evals)):
        load = "  ".join(f"{LABEL[o]}:{evecs[j, i]:+.2f}"
                         for j, o in enumerate(OBJECTIVES))
        print(f"  PC{i+1:<4}{evals[i]:>9.3f}{ratio[i]:>10.1%}"
              f"{ratio[:i+1].sum():>9.1%}   {load}")

    eff = effective_dim(evals)
    n90 = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)
    print(f"\n  有效维度（participation ratio）= {eff:.2f} / 4.00")
    print(f"  解释 90% 方差需要 {n90} 个主成分")
    print(f"  -> 名义上 4 个目标，实际的权衡空间约 {eff:.1f} 维。")

    # ---------- 分 editor 的分布：editor 是否真的各管一维 ----------
    print("\n各 editor 产出候选的平均分数（检查 objective-conditioned 是否奏效）")
    print(f"  {'editor':<8}{'n':>7}" + "".join(f"{LABEL[o]:>9}" for o in OBJECTIVES))
    by_editor = {}
    base = X[origin == "seed"].mean(axis=0)
    for e in EXPERTS:
        m = origin == e
        if m.sum() < 10:
            continue
        mu = X[m].mean(axis=0)
        by_editor[e] = {"n": int(m.sum()), "mean": mu.round(4).tolist(),
                        "delta_vs_seed": (mu - base).round(4).tolist()}
        print(f"  {LABEL[e]:<8}{int(m.sum()):>7}"
              + "".join(f"{v:>9.3f}" for v in mu))
    print(f"  {'(种子)':<8}{int((origin=='seed').sum()):>7}"
          + "".join(f"{v:>9.3f}" for v in base))
    print("\n  每个 editor 应该在自己那一列上相对种子最高；"
          "\n  若不成立，说明 objective-conditioned editor 没有真正分化。")

    report = {
        "n_molecules": int(X.shape[0]),
        "n_seeds": int((origin == "seed").sum()),
        "objectives": list(OBJECTIVES),
        "distribution": dist,
        "spearman": S.round(4).tolist(),
        "pca_eigenvalues": evals.round(4).tolist(),
        "pca_variance_ratio": ratio.round(4).tolist(),
        "pca_loadings": evecs.round(4).tolist(),
        "effective_dim": eff,
        "n_components_90pct": n90,
        "by_editor": by_editor,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    np.savez(args.npz, X=X, origin=origin)
    print(f"\n写入 {args.out} 和 {args.npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
