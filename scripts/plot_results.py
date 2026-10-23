"""主结果出图（英文标签，直接进 report）。

  python scripts/plot_results.py --raw results/raw.json --out figures/

生成：
  fig1_hv_vs_step.png       HV 增益 vs editor 调用次数
  fig2_hv_vs_eligible.png   HV 增益 vs 累计合格候选数（控制产出率）
  fig3_actions.png          各 policy 的 expert 调用分布
  fig4_forest.png           配对差 + bootstrap 95% CI
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figstyle import (  # noqa: E402
    DISPLAY, GRID, INK_SECONDARY, OBJ_LABEL, ORDER, PALETTE, apply_style,
    style_for,
)
from src.interfaces import EXPERTS  # noqa: E402


def boot_band(curves, n_boot=2000, seed=0):
    A = np.array(curves)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, A.shape[0], size=(n_boot, A.shape[0]))
    means = A[idx].mean(axis=1)
    return (A.mean(axis=0), np.percentile(means, 2.5, axis=0),
            np.percentile(means, 97.5, axis=0))


def boot_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def end_label(ax, x, y, name) -> None:
    """线末直接标注：图例之外再给一层身份信息，灰度打印也能读。"""
    ax.annotate(DISPLAY.get(name, name), xy=(x, y), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8,
                color=PALETTE.get(name, INK_SECONDARY))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.json")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tag", default="", help="文件名后缀，便于并列多个 run")
    args = ap.parse_args()

    apply_style()
    rows = json.loads(Path(args.raw).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    names = [n for n in ORDER if n in by] + [n for n in by if n not in ORDER]

    # ---------- fig 1: HV vs editor calls ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n in names:
        curves = [r["curve"] for r in by[n]]
        L = min(len(c) for c in curves)
        m, lo, hi = boot_band([c[:L] for c in curves])
        x = np.arange(L)
        ax.plot(x, m, **style_for(n))
        ax.fill_between(x, lo, hi, color=PALETTE.get(n), alpha=0.08, lw=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Editor calls")
    ax.set_ylabel("Hypervolume gain over seed")
    ax.set_title("Hypervolume gain under a matched editor-call budget",
                 loc="left")
    ax.legend(ncol=2)
    ax.grid(alpha=0.5)
    ax.margins(x=0.02)
    fig.text(0.5, -0.04,
             "Shaded bands are marginal bootstrap 95% CIs and overlap heavily; "
             "because episodes are paired,\nthe paired differences in Fig. 5 are "
             "far tighter than this overlap suggests.",
             ha="center", fontsize=8, color=INK_SECONDARY)
    fig.savefig(out / f"fig1_hv_vs_step{tag}.png")
    plt.close(fig)
    print(f"fig1_hv_vs_step{tag}.png")

    # ---------- fig 2: HV vs cumulative eligible candidates ----------
    # 必须在公共支撑集上画：每个 x 处若用"够得着的那些 episode"求均值，
    # 子集随 x 变化，低产出的轨迹被系统性排除，图会得出与配对分析相反的结论。
    if "eligible_curve" in rows[0]:
        from src.analysis import common_support, coverage_at, episode_key

        x_max = 5
        for cand in range(5, 60, 5):
            if coverage_at(rows, cand)["_common"] >= 0.5:
                x_max = cand
            else:
                break
        keys = common_support(rows, x_max)
        grid = np.arange(0, x_max + 1, 1)

        def at_cand(r, xs_grid):
            xs, ys = np.array(r["eligible_curve"]), np.array(r["curve"])
            L = min(len(xs), len(ys))
            return np.interp(xs_grid, xs[:L], ys[:L], left=0.0)

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        ax = axes[0]
        for n in names:
            sub = [r for r in by[n] if episode_key(r) in keys]
            A = np.array([at_cand(r, grid) for r in sub])
            ax.plot(grid, A.mean(axis=0), **style_for(n))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Cumulative eligible candidates")
        ax.set_ylabel("Hypervolume gain over seed")
        ax.set_title(f"(a) Matched candidate budget (n={len(keys)} episodes)",
                     loc="left")
        ax.legend(ncol=2)
        ax.grid(alpha=0.5)
        ax.margins(x=0.02)

        ax = axes[1]
        for a in [n for n in names if n != "random"]:
            va = {episode_key(r): r for r in by[a] if episode_key(r) in keys}
            vb = {episode_key(r): r for r in by["random"] if episode_key(r) in keys}
            ks = sorted(set(va) & set(vb))
            M, LO, HI = [], [], []
            for x in grid:
                d = np.array([at_cand(va[k], [x])[0] - at_cand(vb[k], [x])[0]
                              for k in ks])
                lo, hi = boot_ci(d, n_boot=2000)
                M.append(d.mean()); LO.append(lo); HI.append(hi)
            ax.plot(grid, M, **style_for(a))
            ax.fill_between(grid, LO, HI, color=PALETTE.get(a), alpha=0.10, lw=0)
        ax.axhline(0, color=INK_SECONDARY, lw=1, ls="--")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Cumulative eligible candidates")
        ax.set_ylabel("Paired difference vs Random")
        ax.set_title("(b) Paired differences, same episodes", loc="left")
        ax.legend(ncol=2)
        ax.grid(alpha=0.5)
        ax.margins(x=0.02)
        fig.tight_layout()
        fig.text(0.5, -0.03,
                 "Restricted to episodes in which every policy reaches "
                 f"{x_max} eligible candidates, so all curves describe the same "
                 "episodes.",
                 ha="center", fontsize=8.5, color=INK_SECONDARY)
        fig.savefig(out / f"fig2_hv_vs_eligible{tag}.png")
        plt.close(fig)
        print(f"fig2_hv_vs_eligible{tag}.png")

    # ---------- fig 3: action distribution ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [DISPLAY.get(n, n) for n in names]
    bottom = np.zeros(len(names))
    # 顺序色：四个 expert 是同一类量（调用占比），用单一色相的深浅
    shades = ["#bcd7f4", "#7fb3e8", "#4a8dd8", "#1f5fa8"]
    for j, e in enumerate(EXPERTS):
        vals = np.array([
            sum(r["actions"].count(e) for r in by[n])
            / max(sum(len(r["actions"]) for r in by[n]), 1) for n in names])
        ax.bar(labels, vals, bottom=bottom, label=OBJ_LABEL[e],
               color=shades[j], edgecolor="white", lw=1.2)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.08:   # 只标够大的块，不是每个数都标
                ax.text(i, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="#0b0b0b" if j < 2 else "white")
        bottom += vals
    ax.axhline(0.25, color=INK_SECONDARY, ls=":", lw=1)
    ax.text(len(names) - 0.4, 0.26, "uniform", fontsize=8,
            color=INK_SECONDARY, ha="right")
    ax.set_ylabel("Share of editor calls")
    ax.set_ylim(0, 1)
    ax.set_title("Which editor each policy calls", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              title=None)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.savefig(out / f"fig3_actions{tag}.png")
    plt.close(fig)
    print(f"fig3_actions{tag}.png")

    # ---------- fig 4: forest plot ----------
    def key(r):
        return (r["smiles"], r["episode_seed"])

    final = {n: {key(r): r["final"] for r in by[n]} for n in names}
    comps = [(n, "random") for n in names if n != "random"]
    comps += [("bandit", "greedy"), ("dqn", "bandit")]

    labels, means, los, his = [], [], [], []
    for a, b in comps:
        ks = sorted(set(final[a]) & set(final[b]))
        d = np.array([final[a][k] - final[b][k] for k in ks])
        lo, hi = boot_ci(d)
        labels.append(f"{DISPLAY.get(a,a)} − {DISPLAY.get(b,b)}")
        means.append(d.mean()); los.append(lo); his.append(hi)

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 1.4))
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        sig = lo > 0 or hi < 0
        # 状态色：显著优 / 显著劣 / 不显著，各自配文字标注，不靠颜色单独承载
        c = "#b3261e" if (sig and m < 0) else ("#1baf7a" if sig else "#8a8a85")
        ax.plot([lo, hi], [y[i], y[i]], color=c, lw=2,
                solid_capstyle="round")
        ax.plot(m, y[i], "o", color=c, ms=6, mec="white", mew=1.2)
        ax.text(hi, y[i], "  significant" if sig else "  n.s.", fontsize=8,
                va="center", color=c)
    ax.axvline(0, color=INK_SECONDARY, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Paired difference in final hypervolume gain")
    ax.set_title("Paired comparisons (bootstrap 95% CI)", loc="left")
    ax.grid(axis="x", alpha=0.5)
    ax.margins(x=0.18)
    fig.savefig(out / f"fig4_forest{tag}.png")
    plt.close(fig)
    print(f"fig4_forest{tag}.png")

    # ---------- fig 5: paired difference vs budget ----------
    # 这张才是主图：配对设计下，差异的 CI 比 fig1 里边际 CI 的重叠窄得多，
    # 而且能看出"学出来的 router 只在早期预算下领先，之后被饱和稀释"。
    def curve_at(r, x):
        c = r["curve"]
        return c[min(x, len(c) - 1)]

    L = min(len(r["curve"]) for r in rows)
    xs = np.arange(1, L)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for a in [n for n in names if n != "random"]:
        va = {key(r): r for r in by[a]}
        vb = {key(r): r for r in by["random"]}
        ks = sorted(set(va) & set(vb))
        M, LO, HI = [], [], []
        for x in xs:
            d = np.array([curve_at(va[k], x) - curve_at(vb[k], x) for k in ks])
            lo, hi = boot_ci(d, n_boot=2000)
            M.append(d.mean()); LO.append(lo); HI.append(hi)
        ax.plot(xs, M, **style_for(a))
        ax.fill_between(xs, LO, HI, color=PALETTE.get(a), alpha=0.10, lw=0)
    ax.axhline(0, color=INK_SECONDARY, lw=1, ls="--")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Editor calls")
    ax.set_ylabel("Paired difference vs Random")
    ax.set_title("Learned routers lead only at small budgets", loc="left")
    ax.legend(ncol=2)
    ax.grid(alpha=0.5)
    ax.margins(x=0.02)
    fig.text(0.5, -0.04,
             "Paired per-episode differences with bootstrap 95% CIs. "
             "Above zero = better than Random.",
             ha="center", fontsize=8, color=INK_SECONDARY)
    fig.savefig(out / f"fig5_paired_vs_budget{tag}.png")
    plt.close(fig)
    print(f"fig5_paired_vs_budget{tag}.png")

    print(f"\n写入 {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
