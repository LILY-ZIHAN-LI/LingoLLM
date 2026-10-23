"""routing 收益 vs action 空间有效维度：把 §11 的两个点做成一条曲线。

为什么需要这个脚本而不是直接读各 run 的日志：
  1. 每个 run 自己挑的候选预算 x 不一样（results_base0 挑了 10，
     seed1/seed2 挑了 5），各变体之间就不可比。这里**固定 x** 重算。
  2. 每个变体有 3 个训练种子。§7 测得重训噪声 0.00355 比 §11a 里所有效应
     都大，所以单种子的点不能用；这里报跨种子的均值和全距，
     并且只在**符号一致**时才认为该点上 routing 有收益。

有效维度取按目标归一的那个口径（findings §10d 的更正）：原始口径被
可达变化幅度最大的目标支配。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (                                      # noqa: E402
    common_support, episode_key, hv_at_eligible,
)

# 变体清单：标签 -> (有效维度 json, [各训练种子的 results 目录])
# α=0 用的是已有的 rdkit 支三个种子，没有重跑。
MANIFEST = {
    "overlap 1.00": ("results_sweep/action_space_a100.json",
                     ["results_sweep/a100_s0", "results_sweep/a100_s1",
                      "results_sweep/a100_s2"]),
    "overlap 0.50": ("results_sweep/action_space_a050.json",
                     ["results_sweep/a050_s0", "results_sweep/a050_s1",
                      "results_sweep/a050_s2"]),
    "overlap 0.25": ("results_sweep/action_space_a025.json",
                     ["results_sweep/a025_s0", "results_sweep/a025_s1",
                      "results_sweep/a025_s2"]),
    "qdis + a0.07": ("results_sweep/action_space_q007.json",
                     ["results_sweep/q007_s0", "results_sweep/q007_s1",
                      "results_sweep/q007_s2"]),
    "qdis + a0.03": ("results_sweep/action_space_q003.json",
                     ["results_sweep/q003_s0", "results_sweep/q003_s1",
                      "results_sweep/q003_s2"]),
    "overlap 0 (base)": ("results_sweep/action_space_a000.json",
                         ["results_base0", "results_seed1", "results_seed2"]),
    "qed disjoint": ("results_sweep/action_space_qdis.json",
                     ["results_sweep/qdis_s0", "results_sweep/qdis_s1",
                      "results_sweep/qdis_s2"]),
    # 生成式支：editor 家族不同，所以在图里要单独标出来，不能当同一族的点
    "generative": ("results_gen/conditioning_delta.json",
                   ["results_gen_router"]),
}
POLICIES = ["round_robin", "gap", "greedy", "bandit", "dqn"]


def eff_dim(path: str) -> float | None:
    """两个诊断脚本写出的 json 用同一个键名（src/analysis.py 的共享实现）。"""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("effective_action_dim_scaled")


def gain_vs_random(rows: list[dict], x: float) -> dict:
    """固定候选预算 x 下，每个 policy 相对 random 的配对差 + bootstrap CI。"""
    keys = common_support(rows, x)
    if not keys:
        return {}
    at = {}
    for r in rows:
        k = episode_key(r)
        if k in keys:
            v = hv_at_eligible(r, x)
            if v is not None:
                at.setdefault(r["policy"], {})[k] = v
    if "random" not in at:
        return {}
    out = {"_n": len(keys)}
    rng = np.random.default_rng(0)
    for name in POLICIES:
        if name not in at:
            continue
        shared = sorted(set(at[name]) & set(at["random"]))
        if not shared:
            continue
        d = np.array([at[name][k] - at["random"][k] for k in shared])
        idx = rng.integers(0, len(d), size=(10000, len(d)))
        m = d[idx].mean(axis=1)
        out[name] = (float(d.mean()),
                     float(np.percentile(m, 2.5)),
                     float(np.percentile(m, 97.5)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-x", "--budget", type=float, default=5.0,
                    help="固定候选预算。5 在各 run 上的公共支撑都有 ~80%%")
    ap.add_argument("--out", default="results_sweep/dimension_curve.json")
    args = ap.parse_args()

    table = {}
    for label, (dim_path, dirs) in MANIFEST.items():
        ed = eff_dim(dim_path)
        per_seed = []
        for d in dirs:
            f = Path(d, "raw.json")
            if not f.exists():
                continue
            per_seed.append(gain_vs_random(json.loads(f.read_text()),
                                           args.budget))
        if not per_seed:
            print(f"!! {label}: 还没有结果，跳过")
            continue
        table[label] = {"eff_dim": ed, "n_seeds": len(per_seed),
                        "seeds": per_seed}

    print(f"\n固定候选预算 x = {args.budget:g}（所有变体同一个 x，才可比）")
    print("单位是终点 HV 增益相对 random 的配对差；* = 该种子的 95% CI 不跨 0\n")
    hdr = f"{'变体':<18}{'eff dim':>8}{'种子':>5}  "
    hdr += "".join(f"{p:>22}" for p in ["greedy", "bandit", "dqn"])
    print(hdr)
    print("-" * len(hdr))
    for label, rec in table.items():
        ed = rec["eff_dim"]
        eds = f"{ed:.2f}" if ed is not None else "  ? "
        cells = []
        for p in ["greedy", "bandit", "dqn"]:
            vals = [s[p] for s in rec["seeds"] if p in s]
            if not vals:
                cells.append(f"{'—':>22}")
                continue
            mu = np.mean([v[0] for v in vals])
            sig = sum(1 for v in vals if v[1] > 0 or v[2] < 0)
            # 符号一致性才是 §7 认可的判据，不是单个 CI
            same = all(v[0] > 0 for v in vals) or all(v[0] < 0 for v in vals)
            mark = ("+" if mu > 0 else "-") if same else "~"
            cells.append(f"{mu:>+12.5f} {mark}{sig}/{len(vals)}      "[:22])
        print(f"{label:<18}{eds:>8}{rec['n_seeds']:>5}  " + "".join(cells))

    print("\n  列里的 +/-/~ 是跨种子的符号一致性（~ = 符号不一致，该点无结论），")
    print("  n/m 是 95% CI 不跨 0 的种子数。§7 的判据是前者，不是后者。")
    print("  'generative' 行是另一个 editor 家族，只有 1 个种子，"
          "在曲线里是参照点而不是同族的点。")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"budget": args.budget, "table": table}, indent=1))
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
