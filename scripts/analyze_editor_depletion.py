"""editor 候选池是否会被"抽干"：连续调用同一个 editor 的产出率随次数怎么变。

动机：DQN 塌缩成 68% gsk3b，而 gsk3b 是高产 editor（约 100% 合格率），
按理它应该拿到**比 random 更多**的候选。但实测 DQN 平均只有 45.5 个合格候选，
比 random 的 53.9 还少。这说明产出率不是 editor 的固定属性，
而是随"连续调用次数"衰减的——editor 是确定性的，同一个分子上反复调用
同一个 editor 只会反复提出同一批编辑。

如果成立，这给"路由为什么重要"一个与目标权衡无关的机制：
轮换 editor 的价值在于**避免提案枯竭**，而不是在目标之间取舍。

只用 results/raw.json，不重跑实验。
  python scripts/analyze_editor_depletion.py --raw results/raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPERTS = ("gsk3b", "jnk3", "qed", "sa")
ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def per_step_eligible(r: dict) -> np.ndarray:
    """eligible_curve 是累计值，差分还原每一步的合格候选数。"""
    return np.diff(np.asarray(r["eligible_curve"], dtype=float))


def run_lengths(actions: list[str]) -> list[int]:
    """第 t 步时，当前 expert 已经连续被调用到第几次（首次为 1）。"""
    out, run = [], 0
    prev = None
    for a in actions:
        run = run + 1 if a == prev else 1
        out.append(run)
        prev = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.json")
    ap.add_argument("--out", default="results/editor_depletion.json")
    ap.add_argument("--max-run", type=int, default=6)
    args = ap.parse_args()

    rows = json.loads(Path(args.raw).read_text())
    names = [n for n in ORDER if any(r["policy"] == n for r in rows)]

    # (expert, run-length) -> 该步的合格候选数
    by_run: dict = defaultdict(list)
    by_expert_run: dict = defaultdict(list)
    # 控制 step 位置：产出率也可能只是随 t 衰减（分子越编越难编）
    by_step: dict = defaultdict(list)
    for r in rows:
        el = per_step_eligible(r)
        acts = r["actions"]
        runs = run_lengths(acts)
        for t, (a, run) in enumerate(zip(acts, runs)):
            if t >= len(el):
                break
            by_run[min(run, args.max_run)].append(el[t])
            by_expert_run[(a, min(run, args.max_run))].append(el[t])
            by_step[t].append(el[t])

    k = 4  # k_per_call
    print(f"{args.raw}: {len(rows)} episodes\n")
    print("每步合格候选数 vs 该 expert 已连续调用次数（k=4 个提案/次）")
    print(f"  {'连续第':>6}{'合格候选':>10}{'合格率':>9}{'样本数':>9}")
    rows_run = {}
    for run in sorted(by_run):
        v = np.array(by_run[run])
        rows_run[run] = {"mean": float(v.mean()), "n": int(len(v))}
        print(f"  {run:>6}{v.mean():>10.2f}{v.mean()/k:>9.0%}{len(v):>9}")

    first = rows_run[1]["mean"]
    last = rows_run[max(rows_run)]["mean"]
    print(f"\n  首次调用 {first:.2f} -> 连续第 {max(rows_run)} 次 {last:.2f}"
          f"（{(last/max(first,1e-9)-1):+.0%}）")

    print("\n对照：每步合格候选数 vs step 位置（排除『只是越到后面越难编』）")
    print(f"  {'step':>6}{'合格候选':>10}{'样本数':>9}")
    step_means = {}
    for t in sorted(by_step)[:args.max_run + 2]:
        v = np.array(by_step[t])
        step_means[t] = float(v.mean())
        print(f"  {t:>6}{v.mean():>10.2f}{len(v):>9}")

    # ---- 控制"卡住"这个混淆 ----
    # env 的 transition rule：只要有合格候选，current 就换成增益最大的那个。
    # 所以"上一步有合格候选" == "分子确实移动了"。
    # 如果分子移动后产出率仍随连续次数下降，才是真的池子枯竭；
    # 如果移动后就恢复，那下降只是"确定性 policy 在死状态上空转"的选择效应。
    moved: dict = defaultdict(list)
    stuck: dict = defaultdict(list)
    for r in rows:
        el = per_step_eligible(r)
        runs = run_lengths(r["actions"])
        for t in range(1, min(len(el), len(runs))):
            bucket = moved if el[t - 1] > 0 else stuck
            bucket[min(runs[t], args.max_run)].append(el[t])

    print("\n控制『分子有没有移动』（上一步是否产出合格候选）")
    print(f"  {'连续第':>6}{'移动后':>10}{'n':>8}{'卡住':>10}{'n':>8}")
    ctrl = {}
    for run in sorted(set(moved) | set(stuck)):
        a = np.array(moved.get(run, []))
        b = np.array(stuck.get(run, []))
        ctrl[run] = {"moved": float(a.mean()) if len(a) else None,
                     "n_moved": int(len(a)),
                     "stuck": float(b.mean()) if len(b) else None,
                     "n_stuck": int(len(b))}
        am = f"{a.mean():.2f}" if len(a) else "·"
        bm = f"{b.mean():.2f}" if len(b) else "·"
        print(f"  {run:>6}{am:>10}{len(a):>8}{bm:>10}{len(b):>8}")
    mv = [ctrl[r]["moved"] for r in sorted(ctrl)
          if ctrl[r]["moved"] is not None and ctrl[r]["n_moved"] >= 30]
    if len(mv) >= 2:
        print(f"\n  仅看分子移动过的步：{mv[0]:.2f} -> {mv[-1]:.2f}"
              f"（{(mv[-1]/max(mv[0],1e-9)-1):+.0%}）")
        print("  这一列若仍明显下降 = 真实的提案池枯竭；"
              "\n  若基本持平 = 下降主要来自确定性 policy 卡在死状态上空转。")

    print("\n分 expert 看（行=expert，列=连续第几次调用）")
    header = "".join(f"{r:>9}" for r in sorted(by_run))
    print(f"  {'expert':<8}{header}")
    per_expert = {}
    for e in EXPERTS:
        cells, vals = [], {}
        for run in sorted(by_run):
            v = by_expert_run.get((e, run), [])
            if len(v) < 20:
                cells.append(f"{'·':>9}")
                continue
            m = float(np.mean(v))
            vals[run] = {"mean": m, "n": len(v)}
            cells.append(f"{m:>9.2f}")
        per_expert[e] = vals
        print(f"  {e:<8}" + "".join(cells))

    # 各 policy 的"重复调用"程度 vs 它拿到的候选总数
    print("\n各 policy 的连续重复程度与总产出")
    print(f"  {'policy':<13}{'平均run长':>10}{'重复步占比':>11}{'终点候选':>10}")
    pol = {}
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        runs = [run_lengths(r["actions"]) for r in sub]
        mean_run = float(np.mean([np.mean(x) for x in runs]))
        rep = float(np.mean([np.mean([v > 1 for v in x]) for x in runs]))
        el = float(np.mean([r["eligible_curve"][-1] for r in sub]))
        pol[n] = {"mean_run": mean_run, "frac_repeat": rep, "eligible": el}
        print(f"  {n:<13}{mean_run:>10.2f}{rep:>11.0%}{el:>10.1f}")

    # ---- 死步预算核算 ----
    # 上面的控制说明：一步产出 0 个合格候选 -> 分子不动 -> 确定性 policy
    # 重选同一个 editor -> 又是 0。这是个吸收态。代价直接用预算算出来。
    print("\n死步预算核算（死步 = 该步 0 个合格候选，消耗调用但零产出）")
    print(f"  {'policy':<13}{'死步占比':>9}{'其中重复上一步':>14}"
          f"{'首次死步@':>10}{'死后还剩':>9}")
    dead = {}
    for n in names:
        sub = [r for r in rows if r["policy"] == n]
        fr, rep_fr, first, tail = [], [], [], []
        for r in sub:
            el = per_step_eligible(r)
            acts = r["actions"][: len(el)]
            d = el == 0
            fr.append(d.mean())
            idx = [t for t in range(1, len(el)) if d[t]]
            if idx:
                rep_fr.append(np.mean([acts[t] == acts[t - 1] for t in idx]))
            if d.any():
                f = int(np.argmax(d))
                first.append(f)
                tail.append(len(el) - f)
        dead[n] = {"frac_dead": float(np.mean(fr)),
                   "frac_dead_repeating": float(np.mean(rep_fr)) if rep_fr else None,
                   "first_dead_step": float(np.mean(first)) if first else None,
                   "steps_after_first_dead": float(np.mean(tail)) if tail else None}
        rp = f"{np.mean(rep_fr):.0%}" if rep_fr else "·"
        fs = f"{np.mean(first):.1f}" if first else "·"
        tl = f"{np.mean(tail):.1f}" if tail else "·"
        print(f"  {n:<13}{np.mean(fr):>9.0%}{rp:>14}{fs:>10}{tl:>9}")
    print("  『其中重复上一步』高 = 死步主要是在同一个 editor 上空转，"
          "\n  而不是分散的偶发失败——这正是吸收态的指纹。")

    x = np.array([pol[n]["frac_repeat"] for n in names])
    y = np.array([pol[n]["eligible"] for n in names])
    if len(names) > 2:
        c = float(np.corrcoef(x, y)[0, 1])
        print(f"\n  重复步占比 与 终点候选数 的相关：r = {c:+.2f}"
              f"（{len(names)} 个 policy）")

    Path(args.out).write_text(json.dumps(
        {"by_run": rows_run, "by_step": step_means, "moved_vs_stuck": ctrl,
         "by_expert_run": {e: {str(r): v for r, v in d.items()}
                           for e, d in per_expert.items()},
         "by_policy": pol, "dead_steps": dead}, indent=2))
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
