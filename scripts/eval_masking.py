"""吸收态的代价有多大：评估时加一条"重试规则"，不重训。

规则很小：如果 router 选的 editor 已经在当前分子上产出过 0 个合格候选，
就改选下一个还没失败过的 editor（按固定顺序，保持确定性）。
别的一切不变——同样的 editor、同样的打分、同样的 transition rule、
同样 20 次调用的预算。

这给出的是机制代价的**上界估计**：它不需要学任何东西，
所以 masked 与 unmasked 的差就是"光是不再对着死 editor 空转"能换回多少 HV。
跟 escape-bit 那组（把同样的信息喂进观测、让策略自己学）互为对照：
一个是结构性修复，一个是表示性修复。

  python scripts/eval_masking.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import bootstrap_ci  # noqa: E402
from src.env import MolEnv  # noqa: E402
from src.interfaces import EXPERTS, N_ACTIONS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402


def rollout_masked(env, policy, smiles, episode_seed, mask: bool):
    obs = env.reset(smiles, episode_seed)
    if hasattr(policy, "reset"):
        policy.reset(episode_seed)
    hv0 = env.hv
    curve, actions, n_overridden = [0.0], [], 0
    for t in range(env.cfg.max_steps):
        a = policy.act(obs, t)
        if mask and a in env.failed_on_current:
            # 固定顺序找下一个没在当前分子上失败过的 editor；
            # 全都失败过就维持原选择（这时无论选谁都是死步）
            for d in range(1, N_ACTIONS):
                alt = (a + d) % N_ACTIONS
                if alt not in env.failed_on_current:
                    a = alt
                    n_overridden += 1
                    break
        obs, _, done, info = env.step(a)
        actions.append(EXPERTS[a])
        curve.append(info["hv"] - hv0)
        if done:
            break
    eligible = [lg.n_scaffold_ok for lg in env.logs]
    return {
        "curve": curve, "final": float(curve[-1]), "actions": actions,
        "eligible": int(sum(eligible)),
        "eligible_curve": np.cumsum([0] + eligible).tolist(),
        "dead_steps": int(sum(1 for e in eligible if e == 0)),
        "overridden": n_overridden,
    }


def _init(cfg_dict, blob):
    global _ENV, _POL
    _ENV = MolEnv(OracleBank(), EpisodeConfig(**cfg_dict))
    _POL = pickle.loads(blob)


def _work(job):
    name, smi, eseed, mask = job
    r = rollout_masked(_ENV, _POL[name], smi, eseed, mask)
    r.update({"policy": name, "smiles": smi, "episode_seed": eseed,
              "mask": mask})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="results/policies.pkl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/masking.json")
    ap.add_argument("--raw-out", default="results/masking_raw.json",
                    help="重试规则下的原始行，可直接喂给 compare_budgets.py")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    policies = pickle.loads(Path(args.policies).read_bytes())
    seeds = Path(args.data).joinpath("seeds_test.txt").read_text().split()
    seeds = seeds[: args.n_test]
    cfg = {"max_steps": args.max_steps, "lam": 0.0}

    jobs = [(n, m, rep * 10000 + i, mask)
            for n in policies for mask in (False, True)
            for rep in range(args.repeats) for i, m in enumerate(seeds)]
    print(f"评估 {len(jobs)} 个 episode（{args.workers} 进程）...")
    with Pool(args.workers, initializer=_init,
              initargs=(cfg, pickle.dumps(policies))) as pool:
        rows = pool.map(_work, jobs, chunksize=4)

    def pick(n, mask, key="final"):
        return {(r["smiles"], r["episode_seed"]): r[key] for r in rows
                if r["policy"] == n and r["mask"] == mask}

    print(f"\n{'policy':13}{'原始HV':>9}{'重试后HV':>10}{'配对差':>10}"
          f"{'95% CI':>22}{'死步':>13}{'改选占比':>9}")
    report = {}
    for n in policies:
        a, b = pick(n, True), pick(n, False)
        ks = sorted(set(a) & set(b))
        d = np.array([a[k] - b[k] for k in ks])
        lo, hi = bootstrap_ci(d)
        d0 = np.mean([r["dead_steps"] for r in rows
                      if r["policy"] == n and not r["mask"]]) / args.max_steps
        d1 = np.mean([r["dead_steps"] for r in rows
                      if r["policy"] == n and r["mask"]]) / args.max_steps
        ov = np.mean([r["overridden"] for r in rows
                      if r["policy"] == n and r["mask"]]) / args.max_steps
        sig = "*" if (lo > 0 or hi < 0) else " "
        report[n] = {"base": float(np.mean(list(b.values()))),
                     "masked": float(np.mean(list(a.values()))),
                     "diff": float(d.mean()), "lo": lo, "hi": hi,
                     "dead_base": float(d0), "dead_masked": float(d1),
                     "override_rate": float(ov), "n": len(ks)}
        print(f"{n:13}{np.mean(list(b.values())):>9.4f}"
              f"{np.mean(list(a.values())):>10.4f}{d.mean():>+10.5f}"
              f"  [{lo:+.5f}, {hi:+.5f}]{sig}"
              f"{d0:>7.0%}->{d1:<5.0%}{ov:>9.0%}")

    print("\n  『改选占比』= 有多少比例的步被重试规则改掉了原本的选择。")
    print("  配对差显著为正 = 吸收态确实在吃掉 HV，而且不需要学任何东西就能拿回来。")

    # 重试规则之后，学出来的 router 还赢 random 吗
    print(f"\n重试规则下 vs random（都加了重试）")
    base = pick("random", True)
    for n in policies:
        if n == "random":
            continue
        cur = pick(n, True)
        ks = sorted(set(base) & set(cur))
        d = np.array([cur[k] - base[k] for k in ks])
        lo, hi = bootstrap_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        report.setdefault("vs_random_masked", {})[n] = {
            "mean": float(d.mean()), "lo": lo, "hi": hi}
        print(f"  {n:13}{d.mean():>+10.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    # 重试规则让策略拿到更多候选，所以『按 editor 调用』的结论必须在
    # 『按候选数』口径下也成立才站得住。存成 raw.json 的格式，
    # 直接 compare_budgets.py --raw results/masking_raw.json 交叉验证。
    Path(args.raw_out).write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "mask"}
         for r in rows if r["mask"]]))
    print(f"写入 {args.raw_out}（仅重试规则下的行）")
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
