"""两种预算口径的并列比较。可以直接跑在已有的 results/raw.json 上，不用重跑实验。

  python scripts/compare_budgets.py --raw results/raw.json
  python scripts/compare_budgets.py --raw results_s8/raw.json --cand-budget 20

为什么需要这个脚本：
  表里的「HV/候选」比值系统性偏袒候选少的策略（HV 增益饱和，分母越小比值越大），
  不能用来下结论。干净的做法是固定候选预算 x，比较各 policy 在**同一个 x** 处的 HV。
  这个脚本就是做这件事，并把 editor 调用口径的同一组比较并排打出来。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    common_support, coverage_at, episode_key, hv_at_eligible, hv_at_step,
    pick_candidate_budget, print_pair_table, summarize_pair,
)

ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.json")
    ap.add_argument("--cand-budget", type=int, nargs="*", default=None,
                    help="候选预算点，可给多个；缺省自动选一个保守点加几个常用点")
    ap.add_argument("--min-common", type=float, default=0.8,
                    help="候选预算点上要求的公共支撑覆盖率")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="要比较的 editor 调用预算点")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = json.loads(Path(args.raw).read_text())
    names = [n for n in ORDER if any(r["policy"] == n for r in rows)]
    max_steps = max(len(r["curve"]) for r in rows) - 1
    steps = args.steps or sorted({3, 5, max_steps // 2, max_steps})

    reach = np.array([r["eligible_curve"][-1] for r in rows])
    x_list = args.cand_budget or sorted(
        {pick_candidate_budget(rows, args.min_common), 10, 20, 30})
    x_list = [x for x in x_list if x > 0]
    # 水平表用最保守的那个点（公共支撑最大）
    x_cand = x_list[0]
    keys = common_support(rows, x_cand)
    cov = coverage_at(rows, x_cand)

    print(f"{args.raw}:  {len(rows)} episodes, {max_steps} steps, "
          f"{len(names)} policies")
    print(f"终点合格候选数：中位 {np.median(reach):.0f}，"
          f"25 分位 {np.percentile(reach, 25):.0f}，最小 {reach.min():.0f}")
    print(f"候选预算点 x = {x_cand}，公共支撑 {len(keys)} 个 episode "
          f"（{cov['_common']:.0%}）")
    print("  各 policy 够得着 x 的比例： " + "  ".join(
        f"{n}:{cov[n]:.0%}" for n in names))
    print("  公共支撑 = 所有 policy 都攒够 x 个候选的 episode；"
          "所有候选口径的比较都只用这批，避免丢弃低产出轨迹造成的选择偏差。\n")

    # ---------- 各 policy 在两种口径下的水平 ----------
    print(f"{'policy':13}{'终点HV':>9}{'终点候选':>9}"
          f"{'HV@x=' + str(x_cand):>11}{'够得着':>8}")
    levels = {}
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        fin = np.mean([r["final"] for r in sub])
        el = np.mean([r["eligible_curve"][-1] for r in sub])
        at = [hv_at_eligible(r, x_cand) for r in sub
              if episode_key(r) in keys]
        ok = [v for v in at if v is not None]
        levels[n] = {"final": float(fin), "eligible": float(el),
                     "hv_at_cand": float(np.mean(ok)) if ok else None,
                     "frac_reaching": float(cov[n])}
        hv_s = f"{np.mean(ok):.4f}" if ok else "  n/a"
        print(f"{n:13}{fin:>9.4f}{el:>9.1f}{hv_s:>11}{cov[n]:>8.0%}")

    print("\n  『终点HV』和『HV@x』排序不一致 = 预算口径翻转。"
          "\n  前者按 editor 调用计费，偏袒高产 editor；后者按候选分子计费。")

    pairs = [(n, "random") for n in names if n != "random"]
    pairs += [("bandit", "greedy"), ("dqn", "bandit"), ("greedy", "round_robin")]
    pairs = [(a, b) for a, b in pairs if a in names and b in names]

    report = {"raw": args.raw, "x_cand": x_cand, "coverage": cov,
              "n_common_support": len(keys), "levels": levels,
              "by_step": {}, "by_candidate": None}

    for x in steps:
        res = [summarize_pair(rows, a, b, lambda r: hv_at_step(r, x))
               for a, b in pairs]
        print_pair_table(res, f"[editor 调用口径] 配对比较 @ 第 {x} 次调用")
        report["by_step"][str(x)] = res

    report["by_candidate"] = {}
    print("\n候选预算点的覆盖率权衡（预算越大区分度越高，但够得着的 episode 越少）：")
    print(f"  {'x':>5}  {'公共支撑':>8}  " + "  ".join(f"{n:>11}" for n in names))
    for x in x_list:
        c = coverage_at(rows, x)
        print(f"  {x:>5}  {c['_common']:>8.0%}  "
              + "  ".join(f"{c[n]:>11.0%}" for n in names))
    for x in x_list:
        k = common_support(rows, x)
        if len(k) < 30:
            print(f"\n[候选口径] x={x} 跳过：公共支撑只剩 {len(k)} 个 episode")
            continue
        res = [summarize_pair(rows, a, b, lambda r: hv_at_eligible(r, x),
                              restrict=k) for a, b in pairs]
        print_pair_table(res, f"[候选口径] 配对比较 @ 累计 {x} 个合格候选"
                              f"（公共支撑 n={len(k)}）")
        report["by_candidate"][str(x)] = res

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
