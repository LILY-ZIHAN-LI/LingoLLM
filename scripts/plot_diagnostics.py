"""诊断图（英文标签）：吸收态、目标相关性、action space 退化。

依赖先跑过这三个分析脚本的 json 输出：
  analyze_editor_depletion.py -> results/editor_depletion.json
  analyze_objectives.py       -> results/objective_analysis.json
  analyze_action_space.py     -> results/action_space.json

  python scripts/plot_diagnostics.py

生成：
  fig7_absorbing.png     死步 / 吸收态
  fig8_objectives.png    目标两两相关 + PCA
  fig9_action_space.png  editor 相似度 + 专精度检验
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figstyle import (  # noqa: E402
    DISPLAY, INK_SECONDARY, OBJ_LABEL, ORDER, PALETTE, apply_style,
)
from src.interfaces import EXPERTS, OBJECTIVES  # noqa: E402

# 发散色板：两个色相 + 中性灰中点（绝不在中点放色相）
DIVERGING = LinearSegmentedColormap.from_list(
    "bo", ["#2a78d6", "#8ab4e8", "#e8e8e5", "#f0a583", "#eb6834"])
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#eef4fb", "#bcd7f4", "#7fb3e8", "#4a8dd8", "#1f5fa8"])


def heatmap(ax, M, labels, cmap, vmin, vmax, fmt="{:+.2f}"):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i][j]
            # 文字用黑/白保证对比，不用序列色
            shade = abs(v - (vmin + vmax) / 2) / max((vmax - vmin) / 2, 1e-9)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                    color="white" if shade > 0.6 else "#0b0b0b")
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_absorbing(dep, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # (a) 产出率 vs 连续调用次数，按"分子是否移动"分开
    ax = axes[0]
    ctrl = dep["moved_vs_stuck"]
    runs = sorted(int(k) for k in ctrl)
    moved = [ctrl[str(r)]["moved"] for r in runs]
    stuck = [ctrl[str(r)]["stuck"] for r in runs]
    ax.plot(runs, moved, "-o", color="#1baf7a", ms=7, mec="white", mew=1.2,
            label="Molecule moved last step")
    ax.plot(runs, stuck, "--s", color="#b3261e", ms=7, mec="white", mew=1.2,
            label="Molecule stuck last step")
    ax.annotate("no depletion", xy=(runs[-1], moved[-1]), xytext=(-6, 10),
                textcoords="offset points", fontsize=8.5, color="#1baf7a",
                ha="right")
    ax.annotate("exactly zero, forever", xy=(runs[-1], stuck[-1]),
                xytext=(-6, 10), textcoords="offset points", fontsize=8.5,
                color="#b3261e", ha="right")
    ax.set_xlabel("Consecutive calls to the same editor")
    ax.set_ylabel("Eligible candidates per call")
    ax.set_title("(a) Zero yield is an absorbing state, not depletion",
                 loc="left")
    ax.set_ylim(-0.2, 3.6)
    ax.legend(loc="center left")
    ax.grid(alpha=0.5)

    # (b) 各 policy 的死步预算占比
    ax = axes[1]
    dead = dep["dead_steps"]
    names = [n for n in ORDER if n in dead]
    vals = [dead[n]["frac_dead"] for n in names]
    reps = [dead[n]["frac_dead_repeating"] or 0.0 for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    reps = [reps[i] for i in order]
    y = np.arange(len(names))
    ax.barh(y, vals, color=[PALETTE.get(n) for n in names], height=0.62)
    for i, (v, r) in enumerate(zip(vals, reps)):
        ax.text(v + 0.012, i, f"{v:.0%}   ({r:.0%} repeat the same editor)",
                va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(y, [DISPLAY.get(n, n) for n in names], fontsize=9)
    ax.set_xlabel("Share of editor calls that yield nothing")
    ax.set_xlim(0, max(vals) * 1.85)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_title("(b) Deterministic routers burn budget on dead calls",
                 loc="left")
    ax.grid(axis="x", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig7_absorbing.png")
    plt.close(fig)
    print("fig7_absorbing.png")


def fig_objectives(obj, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    labels = [OBJ_LABEL[o] for o in obj["objectives"]]

    ax = axes[0]
    S = np.array(obj["spearman"])
    im = heatmap(ax, S, labels, DIVERGING, -1, 1)
    ax.set_title("(a) Pairwise Spearman correlation", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)

    ax = axes[1]
    ratio = np.array(obj["pca_variance_ratio"])
    x = np.arange(1, len(ratio) + 1)
    ax.bar(x, ratio, color="#7fb3e8", width=0.6, label="Variance explained")
    ax.plot(x, np.cumsum(ratio), "-o", color="#1f5fa8", ms=6, mec="white",
            mew=1.2, label="Cumulative")
    for xi, v in zip(x, np.cumsum(ratio)):
        ax.text(xi, v + 0.04, f"{v:.0%}", ha="center", fontsize=8,
                color=INK_SECONDARY)
    ax.axhline(1.0, color=INK_SECONDARY, lw=0.8, ls=":")
    ax.set_xticks(x, [f"PC{i}" for i in x])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Share of variance")
    ax.set_title(f"(b) PCA: effective dimensionality "
                 f"{obj['effective_dim']:.2f} / 4", loc="left")
    ax.legend(loc="center right")
    ax.grid(axis="y", alpha=0.5)
    fig.tight_layout()
    fig.text(0.5, -0.03,
             "The four objectives do NOT collapse: 90% of variance needs all "
             f"{obj['n_components_90pct']} components.",
             ha="center", fontsize=8.5, color=INK_SECONDARY)
    fig.savefig(out / "fig8_objectives.png")
    plt.close(fig)
    print("fig8_objectives.png")


def fig_action_space(act, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = [OBJ_LABEL[e] for e in EXPERTS]

    ax = axes[0]
    C = np.array(act["editor_cosine"])
    im = heatmap(ax, C, labels, DIVERGING, -1, 1)
    ax.set_xlabel("Editor")
    ax.set_ylabel("Editor")
    ax.set_title(f"(a) Editors collapse into two groups "
                 f"({act['effective_action_dim']:.2f} / 4 effective)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)

    # (b) 专精度：editor j 在目标 j 上 vs 最强的其他 editor
    ax = axes[1]
    spec = act["specialization"]
    objs = [o for o in OBJECTIVES if o in spec]
    y = np.arange(len(objs))[::-1]
    for i, o in enumerate(objs):
        s = spec[o]
        sig = s["lo"] > 0 or s["hi"] < 0
        c = ("#1baf7a" if (sig and s["mean"] > 0)
             else "#b3261e" if sig else "#8a8a85")
        ax.plot([s["lo"], s["hi"]], [y[i], y[i]], color=c, lw=2,
                solid_capstyle="round")
        ax.plot(s["mean"], y[i], "o", color=c, ms=7, mec="white", mew=1.2)
        verdict = ("specialised" if (sig and s["mean"] > 0)
                   else "beaten by rival" if sig else "no difference")
        ax.text(s["hi"] + 0.001, y[i], f"  {verdict}", fontsize=8.5,
                va="center", color=c)
    ax.axvline(0, color=INK_SECONDARY, lw=1, ls="--")
    ax.set_yticks(y, [f"{OBJ_LABEL[o]}  vs {OBJ_LABEL[spec[o]['rival']]}"
                      for o in objs], fontsize=9)
    ax.set_xlabel("Best-candidate improvement, own editor − best rival editor")
    ax.set_title(f"(b) Only {act['n_specialized']}/4 editors win on their "
                 "own objective", loc="left")
    ax.grid(axis="x", alpha=0.5)
    ax.margins(x=0.30)
    fig.tight_layout()
    fig.savefig(out / "fig9_action_space.png")
    plt.close(fig)
    print("fig9_action_space.png")


def fig_seed_variance(sv, out: Path) -> None:
    """效应量 vs 两个噪声源。这张图决定了别的图能怎么说话。"""
    dec = sv["decomposition"]
    floor = dec["baseline_floor_abs_max"]
    names = [n for n in ORDER if n in dec and n != "random"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) 每个 policy 的 9 种 (训练种子 x random 实现) 组合 + 噪声地板
    ax = axes[0]
    y = np.arange(len(names))[::-1]
    ax.axvspan(-floor, floor, color="#e8e8e5", zorder=0)
    ax.axvline(0, color=INK_SECONDARY, lw=1, ls="--", zorder=1)
    for i, n in enumerate(names):
        M = np.array(dec[n]["matrix"])
        ok = dec[n]["sign_consistent"]
        c = PALETTE.get(n, "#8a8a85")
        ax.plot([M.min(), M.max()], [y[i], y[i]], color=c, lw=2, zorder=2,
                solid_capstyle="round")
        # 每个组合一个点；边缘留白保证重叠点可数
        ax.plot(M.ravel(), np.full(M.size, y[i]), "o", color=c, ms=6,
                mec="white", mew=1.2, alpha=0.85, zorder=3)
        # 方向性的判语：一致为负 = 稳定地更差，和"效应成立"不是一回事
        verdict = ("consistently better" if (ok and M.min() > 0)
                   else "consistently worse" if ok else "sign flips")
        ax.text(max(M.max(), floor) + 0.00022, y[i], f"  {verdict}",
                va="center", fontsize=8.5,
                color="#b3261e" if not ok else c)
    ax.set_yticks(y, [DISPLAY.get(n, n) for n in names], fontsize=9)
    ax.set_xlabel("Final hypervolume, policy − random (paired)")
    ax.set_title("(a) Only LinUCB is reliably better than random", loc="left")
    # 灰带直接标注，不进图例——图例框会压到最底下那行
    ax.annotate(f"random baseline\nnoise floor (±{floor:.5f})",
                xy=(0, y[0] + 0.42), ha="center", va="bottom", fontsize=8,
                color=INK_SECONDARY)
    ax.grid(axis="x", alpha=0.5)
    ax.margins(x=0.22)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.25)

    # (b) 两个噪声源的大小，和效应量并排
    ax = axes[1]
    x = np.arange(len(names))
    w = 0.38
    learn = [dec[n]["learning_noise"] for n in names]
    eff = [abs(np.array(dec[n]["matrix"]).mean()) for n in names]
    ax.bar(x - w / 2, eff, w, color="#2a78d6", label="|mean effect|")
    ax.bar(x + w / 2, learn, w, color="#eda100", label="Learning noise")
    ax.axhline(floor, color="#b3261e", lw=1.4, ls=":",
               label="Baseline noise floor")
    ax.set_xticks(x, [DISPLAY.get(n, n) for n in names], fontsize=8.5,
                  rotation=20, ha="right")
    ax.set_ylabel("Hypervolume")
    ax.set_title("(b) Retraining noise is the size of the effect", loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.5)

    fig.tight_layout()
    fig.text(0.5, -0.04,
             "Nine points per policy: three training seeds × three random-baseline "
             "realizations. LinUCB and the two heuristics are deterministic, so "
             "their learning noise is exactly zero.",
             ha="center", fontsize=8.5, color=INK_SECONDARY)
    fig.savefig(out / "fig10_seed_variance.png")
    plt.close(fig)
    print("fig10_seed_variance.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    apply_style()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, fn, key in [
        ("editor_depletion.json", fig_absorbing, "dead_steps"),
        ("objective_analysis.json", fig_objectives, "pca_variance_ratio"),
        ("action_space.json", fig_action_space, "editor_cosine"),
        ("seed_variance.json", fig_seed_variance, "decomposition"),
    ]:
        f = res / name
        if not f.exists():
            print(f"跳过 {name}（未找到，先跑对应的分析脚本）")
            continue
        fn(json.loads(f.read_text()), out)
    print(f"\n写入 {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
