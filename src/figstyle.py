"""所有图共用的样式：英文标签、固定配色、灰度可区分。

配色取自已验证的分类色板（固定顺序，不循环）。random 刻意留成灰色虚线：
它是所有配对比较的基准线，不是六个平级序列之一，视觉上也应该这样读。

每个 policy 还绑定一个线型 + marker 作为第二编码——report 可能被灰度打印，
颜色不能是唯一的身份信息。
"""
from __future__ import annotations

# 分类色板的固定槽位（slot 1,2,3,4,5），random 用中性灰
PALETTE = {
    "random":      "#8a8a85",   # 基准线：中性灰
    "round_robin": "#2a78d6",   # slot 1 blue
    "gap":         "#eb6834",   # slot 2 orange
    "greedy":      "#1baf7a",   # slot 3 aqua
    "bandit":      "#eda100",   # slot 4 yellow
    "dqn":         "#e87ba4",   # slot 5 magenta
}

LINESTYLE = {
    "random": (0, (4, 2)), "round_robin": "-", "gap": (0, (1, 1.5)),
    "greedy": "-", "bandit": (0, (5, 1.5, 1, 1.5)), "dqn": (0, (3, 1, 1, 1)),
}

MARKER = {
    "random": None, "round_robin": "o", "gap": "s",
    "greedy": "^", "bandit": "D", "dqn": "v",
}

# 图里出现的英文名（report 是英文的）
DISPLAY = {
    "random": "Random", "round_robin": "Round-robin", "gap": "Target-gap",
    "greedy": "Greedy (1-step)", "bandit": "LinUCB bandit", "dqn": "DQN",
}

OBJ_LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}

ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]

# 中性色：文字和网格用文字色，不用序列色
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d8d7d2"


def apply_style() -> None:
    """统一的 rcParams。瘦线条、克制的网格、无顶右边框。"""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def style_for(name: str) -> dict:
    """一个 policy 的完整画线参数。"""
    return {
        "color": PALETTE.get(name, INK_SECONDARY),
        "linestyle": LINESTYLE.get(name, "-"),
        "label": DISPLAY.get(name, name),
    }
