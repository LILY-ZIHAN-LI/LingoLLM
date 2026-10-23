"""动作空间塌成两组时，routing 的收益还剩多少？

动机：生成式池的四个条件在余弦相似度上塌成两组
（{gsk3b, jnk3} 0.994、{qed, sa} 0.998，组间 0.92-0.95；findings §10c），
RDKit 池是同一对分组但程度轻一些（§1b）。如果 action 实际上只有两组，
那么 random 的 25/25/25/25 本身就覆盖了两组，而任何"挑最好的 editor"
都会在组内集中 —— 集中在一个与自己近乎等价的 editor 上没有收益，
却丢掉了跨组的覆盖。

检验方式（关键是不混杂 policy 身份）：只在 **random 自己的 episode 内部**
看组间均衡度与终点 HV 的关系。random 的动作序列只因 episode_seed 而变，
均衡度是随机波动出来的，所以这是一个 policy 内的干净对比；
而跨 policy 比均衡度会同时比"是哪个 policy"，那就说明不了机制。

只读 raw.json，不调 oracle。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 分组来自实测余弦矩阵，不是按"两个 docking 目标 + 两个药性目标"手分的。
# 两支的分组恰好一致（生成式 §10c / RDKit §1b），所以同一个常量两边都用。
GROUPS = {"gsk3b": 0, "jnk3": 0, "qed": 1, "sa": 1}
POLICIES = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def balance(actions: list[str]) -> float:
    """组间均衡度：1.0 = 两组各一半，0.0 = 全在一组。"""
    if not actions:
        return float("nan")
    g = np.array([GROUPS[a] for a in actions])
    p1 = float((g == 1).mean())
    return 2.0 * min(p1, 1.0 - p1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """自己算，省掉 scipy 依赖。秩相关对 HV 的重尾分布比 Pearson 稳。"""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def boot_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    out = [spearman(x[i], y[i])
           for i in (rng.integers(0, n, size=n) for _ in range(n_boot))]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="results 目录，按 arm 顺序给")
    args = ap.parse_args()

    for d in args.dirs:
        rows = json.loads(Path(d, "raw.json").read_text())
        print(f"\n{'=' * 74}\n{d}\n{'=' * 74}")

        print(f"{'policy':13}{'组间均衡':>9}{'终点HV':>9}"
              f"{'组内集中度':>11}   动作分布(组1/组2)")
        for name in POLICIES:
            rs = [r for r in rows if r["policy"] == name]
            if not rs:
                continue
            b = np.array([balance(r["actions"]) for r in rs])
            hv = np.array([r["final"] for r in rs])
            acts = [a for r in rs for a in r["actions"]]
            g1 = np.mean([GROUPS[a] == 0 for a in acts])
            # 组内集中度：组内两个 editor 分得多不均，1.0 = 全给一个
            in1 = [a for a in acts if GROUPS[a] == 0]
            in2 = [a for a in acts if GROUPS[a] == 1]
            def skew(xs, pair):
                if not xs:
                    return float("nan")
                f = np.mean([x == pair[0] for x in xs])
                return abs(2 * f - 1)
            conc = np.nanmean([skew(in1, ("gsk3b", "jnk3")),
                               skew(in2, ("qed", "sa"))])
            print(f"{name:13}{b.mean():>9.3f}{hv.mean():>9.4f}"
                  f"{conc:>11.3f}   {g1:.0%}/{1 - g1:.0%}")

        # --- 机制检验：random 内部，均衡度 vs 终点 HV ---
        print("\n  policy 内部的均衡度-HV 秩相关（不混杂 policy 身份）:")
        for name in POLICIES:
            rs = [r for r in rows if r["policy"] == name]
            if not rs:
                continue
            b = np.array([balance(r["actions"]) for r in rs])
            hv = np.array([r["final"] for r in rs])
            if np.nanstd(b) < 1e-9:
                print(f"    {name:13}   均衡度无变异（动作序列固定），不可检验")
                continue
            rho = spearman(b, hv)
            lo, hi = boot_ci(b, hv)
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"    {name:13}{rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}"
                  f"  n={len(rs)}")

        print("\n  同上，但控制产出率：均衡度 vs HV/合格候选")
        for name in ["random", "greedy", "bandit", "dqn"]:
            rs = [r for r in rows if r["policy"] == name]
            if not rs:
                continue
            b = np.array([balance(r["actions"]) for r in rs])
            hv = np.array([r["hv_per_eligible"] for r in rs])
            if np.nanstd(b) < 1e-9:
                continue
            rho = spearman(b, hv)
            lo, hi = boot_ci(b, hv)
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"    {name:13}{rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
