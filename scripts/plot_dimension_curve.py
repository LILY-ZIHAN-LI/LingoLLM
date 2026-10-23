"""fig11：routing 收益 vs action 空间有效维度（findings §12d）。

  python scripts/plot_dimension_curve.py

这张图是 §11 第 5 条结论的主证据。要读出来的三件事：
  1. 同一个 editor 家族内部，收益随维度单调上升（7 个点）；
  2. α=1 那一点是**零假设对照** —— 四个 editor 构造上完全相同，
     任何 router 都只能和 random 等价，实测确实落在 0 上；
  3. 生成式支（另一个家族）落在它的维度所预测的位置上。

误差棒是 3 个训练种子的全距，不是置信区间 —— n=3 给不出可信的 CI，
画成 CI 会把它读得比实际更确定。生成式那一点只有 1 个种子，
画的是该种子自己的 bootstrap CI，并在图上标明口径不同。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.figstyle import (  # noqa: E402
    DISPLAY, GRID, INK, INK_SECONDARY, MARKER, PALETTE, apply_style,
)

SHOW = ["greedy", "bandit", "dqn"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results_sweep/dimension_curve.json")
    ap.add_argument("--out", default="figures/fig11_dimension_curve.png")
    args = ap.parse_args()

    blob = json.loads(Path(args.curve).read_text())
    tab = blob["table"]
    fam = {k: v for k, v in tab.items() if k != "generative"}
    gen = tab.get("generative")

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # "与 random 相等" 的参考线。灰色是图元不是序列色（见 figstyle 的说明）。
    ax.axhline(0, color=PALETTE["random"], linestyle=(0, (4, 2)),
               linewidth=1.2, zorder=1)
    ax.annotate("equal to random", xy=(2.57, 0), xytext=(2.57, -0.00012),
                color=INK_SECONDARY, fontsize=8, ha="right")

    xs_all = sorted(v["eff_dim"] for v in fam.values())
    for name in SHOW:
        pts = sorted((v["eff_dim"], [s[name][0] for s in v["seeds"]])
                     for v in fam.values() if name in v["seeds"][0])
        x = np.array([p for p, _ in pts])
        ys = [np.array(v) for _, v in pts]
        mu = np.array([v.mean() for v in ys])
        lo = np.array([v.min() for v in ys])
        hi = np.array([v.max() for v in ys])
        c = PALETTE[name]

        # OLS 拟合：斜率就是"每多一维 action 空间值多少 HV"这个claim本身
        flatx = np.repeat(x, [len(v) for v in ys])
        flaty = np.concatenate(ys)
        b, a = np.polyfit(flatx, flaty, 1)
        gx = np.linspace(min(xs_all) - 0.04, max(xs_all) + 0.04, 50)
        ax.plot(gx, a + b * gx, color=c, linewidth=1.0, alpha=0.45,
                linestyle="-", zorder=2)

        ax.vlines(x, lo, hi, color=c, linewidth=1.6, alpha=0.75, zorder=3)
        ax.plot(x, mu, marker=MARKER[name], markersize=7.5, color=c,
                linewidth=0, markeredgecolor="white", markeredgewidth=1.0,
                label=f"{DISPLAY[name]}  ({b:+.5f}/dim)", zorder=4)
        # 直接标注：色板的对比度 WARN 要求标签，不能只靠颜色
        ax.annotate(DISPLAY[name].split(" (")[0],
                    xy=(x[-1], mu[-1]), xytext=(6, 2),
                    textcoords="offset points", color=c, fontsize=8.5,
                    fontweight="semibold", va="center")

    # 生成式支：另一个 editor 家族 + 只有 1 个种子，所以刻意不用序列色、
    # 不连线，画成空心黑点，避免被读成曲线上的第四个序列。
    if gen is not None and gen["eff_dim"]:
        gx = gen["eff_dim"]
        # 三个 policy 在 x 上各错开一点：否则三条单种子 CI 会叠成一条
        # 贯穿全图的竖线，读起来像分隔线而不是数据。
        for off, name in zip((-0.035, 0.0, 0.035), SHOW):
            if name not in gen["seeds"][0]:
                continue
            m, clo, chi = gen["seeds"][0][name]
            ax.vlines(gx + off, clo, chi, color=INK, linewidth=0.9,
                      alpha=0.30, zorder=3)
            ax.plot([gx + off], [m], marker=MARKER[name], markersize=7.0,
                    markerfacecolor="none", markeredgecolor=INK,
                    markeredgewidth=1.2, linewidth=0, zorder=5)
        ax.annotate("generative backbone\n(other editor family, 1 seed;\n"
                    "bars are that seed's CI)",
                    xy=(gx + 0.05, -0.00075), xytext=(2.03, -0.00155),
                    color=INK, fontsize=8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK_SECONDARY,
                                    linewidth=0.8, shrinkA=4, shrinkB=3))

    # 零假设对照的注解：这一点落在 0 上是对测量管道的检验，不是一个发现
    x0 = min(xs_all)
    # 文字放到右上方、箭头拉长，否则短箭头会看起来指向 1.28 那一簇
    ax.annotate("null control: four editors\nidentical by construction",
                xy=(x0, 0.00006), xytext=(1.16, 0.00128),
                color=INK_SECONDARY, fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK_SECONDARY,
                                linewidth=0.8, shrinkA=4, shrinkB=4))

    ax.set_xlabel("Effective action dimension  "
                  "(participation ratio, per-objective normalised)")
    ax.set_ylabel("Hypervolume gain vs random\n"
                  "(paired, at matched candidate budget x = 5)")
    ax.set_title("Routing gain tracks action-space dimensionality, "
                 "not router sophistication", fontsize=11, loc="left",
                 pad=12)
    ax.set_xlim(0.92, 2.62)
    # 下界必须容得下生成式那一点的单种子 CI（下沿约 -0.0018）：
    # 截断的误差棒会把不确定性画得比实际小。
    ax.set_ylim(-0.00215, 0.00272)
    ax.legend(loc="upper left", ncol=1, handletextpad=0.6)

    cap = ("Error bars are the range over 3 training seeds (not a CI: n = 3). "
           "RDKit pool overlap swept via src/editors.py:_mix; "
           "points placed by measured dimension, not knob setting.")
    fig.text(0.0, -0.045, cap, fontsize=7.5, color=INK_SECONDARY, ha="left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
