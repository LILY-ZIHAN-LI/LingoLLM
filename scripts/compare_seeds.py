"""训练种子方差：同一配置换训练种子重跑，看效应量和噪声谁大。

8 步 ablation 暴露的问题：greedy-random 只因重训一次就从 +0.0008 翻到
-0.0012。所以任何"学出来的 router 更好"的说法都必须带跨种子的带宽，
否则报告的是训练噪声而不是方法的性质。

这个脚本做两件事：
  1. 多个 results 目录（同配置、不同训练种子）逐个算 policy-random 的
     配对差，再给出跨种子的 min/max 带宽；
  2. escape-bit A/B：同种子下 escape 组 vs 对照组的逐 episode 配对差。
     两组跑的是同一批 (smiles, episode_seed)，所以可以直接配对。

  python scripts/compare_seeds.py \
      --runs results_base0 results_seed1 results_seed2 \
      --escape results_escape0:results_base0 results_escape1:results_seed1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import bootstrap_ci, episode_key  # noqa: E402

ORDER = ["random", "round_robin", "gap", "greedy", "bandit", "dqn"]


def load(d: str) -> list[dict]:
    return json.loads((Path(d) / "raw.json").read_text())


def by_key(rows, policy, field="final"):
    return {episode_key(r): r[field] for r in rows if r["policy"] == policy}


def paired(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "lo": lo, "hi": hi, "n": len(ks),
            "sig": lo > 0 or hi < 0}


def seed_band(runs: list[str]) -> dict:
    """每个 policy 在每个训练种子下的 HV 和 (policy - random) 配对差。"""
    per_run = {}
    for d in runs:
        rows = load(d)
        rnd = by_key(rows, "random")
        pol = {}
        for n in ORDER:
            cur = by_key(rows, n)
            pol[n] = {"hv": float(np.mean(list(cur.values()))),
                      "vs_random": paired(cur, rnd) if n != "random" else None}
        per_run[d] = pol

    print(f"\n{'='*78}\n跨训练种子：绝对 HV\n{'='*78}")
    print(f"{'policy':13}" + "".join(f"{Path(d).name:>16}" for d in runs)
          + f"{'spread':>10}")
    for n in ORDER:
        vs = [per_run[d][n]["hv"] for d in runs]
        print(f"{n:13}" + "".join(f"{v:>16.4f}" for v in vs)
              + f"{max(vs)-min(vs):>10.4f}")

    print(f"\n{'='*78}\n跨训练种子：(policy - random) 配对差\n"
          f"  * = 该种子下 95% CI 不跨 0\n{'='*78}")
    print(f"{'policy':13}" + "".join(f"{Path(d).name:>16}" for d in runs)
          + f"{'band':>22}{'符号一致':>10}")
    band = {}
    for n in ORDER:
        if n == "random":
            continue
        ds = [per_run[d][n]["vs_random"] for d in runs]
        ms = [x["mean"] for x in ds]
        cells = "".join(f"{x['mean']:>+15.5f}{'*' if x['sig'] else ' '}"
                        for x in ds)
        same = "是" if (min(ms) > 0 or max(ms) < 0) else "否"
        band[n] = {"per_seed": ms, "lo": min(ms), "hi": max(ms),
                   "sign_consistent": same == "是",
                   "n_sig": sum(x["sig"] for x in ds)}
        print(f"{n:13}{cells}  [{min(ms):+.5f}, {max(ms):+.5f}]{same:>8}")

    print("\n  『band』是跨种子的 min/max，不是 CI。它和单种子 CI 必须一起报：")
    print("  单种子 CI 只刻画 episode 抽样噪声，band 刻画训练噪声。")
    print("  符号不一致的 policy 不能说『比 random 好』——那是种子运气。")
    return {"per_run": per_run, "band": band}


def escape_ab(pairs: list[str]) -> dict:
    """escape-bit 组 vs 同种子对照组，逐 episode 配对。"""
    print(f"\n{'='*78}\nescape-bit A/B（观测里加 4 个『该 editor 在此分子上失败过』位）\n"
          f"{'='*78}")
    out = {}
    for spec in pairs:
        esc_d, base_d = spec.split(":")
        esc, base = load(esc_d), load(base_d)
        print(f"\n{Path(esc_d).name} vs {Path(base_d).name}")
        print(f"  {'policy':13}{'对照':>9}{'escape':>9}{'配对差':>11}"
              f"{'95% CI':>24}")
        for n in ORDER:
            a, b = by_key(esc, n), by_key(base, n)
            if not a or not b:
                continue
            r = paired(a, b)
            print(f"  {n:13}{np.mean(list(b.values())):>9.4f}"
                  f"{np.mean(list(a.values())):>9.4f}{r['mean']:>+11.5f}"
                  f"  [{r['lo']:+.5f}, {r['hi']:+.5f}]"
                  f"{'*' if r['sig'] else ' '}")
            out.setdefault(spec, {})[n] = r
    print("\n  random / round_robin 是阴性对照：它们不读观测里的这几位，")
    print("  所以这两行的差只反映重训噪声，可以当噪声地板读。")
    return out


def decompose(runs: list[str]) -> dict:
    """把 band 拆成两个噪声源。

    --seed 同时喂给了 make_baselines(seed=...)，所以 RandomPolicy 在
    每个 run 里是不同的实现（src/policies.py:33 用 self.seed 参与 reset）。
    于是 (policy_i - random_i) 的跨种子带宽混了两样东西：
      (a) 学习噪声：policy 自己被重训出来的差别；
      (b) 基线噪声：random 这个随机策略本身的实现差别。
    round_robin / gap / bandit 是确定性的（spread 恰为 0.0000），
    它们的 band 只含 (b)；greedy / dqn 两者都有。
    拿 i×j 的全部组合就能分开：固定 j 看 i 的离散度是 (a)，反之是 (b)。
    """
    rows = {d: load(d) for d in runs}
    rnd = {d: by_key(rows[d], "random") for d in runs}

    print(f"\n{'='*78}\n噪声分解：学习噪声 vs random 基线自身的实现噪声\n{'='*78}")
    # 基线噪声地板：random_i vs random_j，两个都只是 random 的不同实现
    floor = [paired(rnd[a], rnd[b])["mean"]
             for i, a in enumerate(runs) for b in runs[i + 1:]]
    print(f"\nrandom_i - random_j（纯基线噪声，没有任何学习）:")
    print("  " + "  ".join(f"{v:+.5f}" for v in floor)
          + f"   |最大| = {max(abs(v) for v in floor):.5f}")

    out = {"baseline_floor": floor,
           "baseline_floor_abs_max": max(abs(v) for v in floor)}
    print(f"\n{'policy':13}{'学习噪声':>12}{'基线噪声':>12}{'全组合 band':>26}"
          f"{'符号一致':>10}")
    for n in ORDER:
        if n == "random":
            continue
        M = np.array([[paired(by_key(rows[i], n), rnd[j])["mean"]
                       for j in runs] for i in runs])
        # 固定一列（同一个 random）看行的离散度 = 学习噪声
        learn = float(np.mean(M.max(axis=0) - M.min(axis=0)))
        # 固定一行（同一个训练好的 policy）看列的离散度 = 基线噪声
        base = float(np.mean(M.max(axis=1) - M.min(axis=1)))
        same = "是" if (M.min() > 0 or M.max() < 0) else "否"
        out[n] = {"learning_noise": learn, "baseline_noise": base,
                  "lo": float(M.min()), "hi": float(M.max()),
                  "sign_consistent": same == "是", "matrix": M.tolist()}
        print(f"{n:13}{learn:>12.5f}{base:>12.5f}"
              f"      [{M.min():+.5f}, {M.max():+.5f}]{same:>10}")

    print("\n  『学习噪声』= 同一个 random 基线下，换训练种子带来的摆动。")
    print("  『基线噪声』= 同一个训练好的 policy，换 random 实现带来的摆动。")
    print("  确定性策略（round_robin / gap / bandit）的学习噪声应当恰为 0。")
    print("  全组合 band 的符号一致才说明效应不是任一种噪声的产物。")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["results_base0", "results_seed1", "results_seed2"])
    ap.add_argument("--escape", nargs="*", default=[],
                    help="escape_dir:control_dir，可给多组")
    ap.add_argument("--out", default="results/seed_variance.json")
    args = ap.parse_args()

    report = {"runs": args.runs}
    report.update(seed_band(args.runs))
    report["decomposition"] = decompose(args.runs)
    if args.escape:
        report["escape_ab"] = escape_ab(args.escape)

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
