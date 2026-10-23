"""主实验：训练 -> 在 held-out seeds 上配对评估。

三类 policy 的训练方式刻意区分开，这样对比才有解释力：
- greedy : 离线拟合 random 轨迹，只看即时 reward，无探索
- bandit : 在线 UCB 探索，只看即时 reward
- dqn    : 在线 epsilon-greedy 探索，看折扣后的未来 reward

greedy -> bandit 的差 = 探索的价值
bandit -> dqn  的差 = 延迟 credit assignment 的价值

用法:
  python scripts/run_experiment.py --quick
  python scripts/run_experiment.py --n-test 200 --repeats 10 --workers 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    common_support, coverage_at, hv_at_eligible, hv_at_step, pick_candidate_budget,
    print_pair_table, summarize_pair,
)
from src.env import MolEnv  # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402
from src.policies import (  # noqa: E402
    DQNPolicy, GapPolicy, GreedyPolicy, LinUCBBandit, make_baselines,
)


# --------------------------------------------------------------------------
def rollout(env, policy, smiles: str, episode_seed: int, record: bool = False):
    obs = env.reset(smiles, episode_seed)
    if hasattr(policy, "reset"):
        policy.reset(episode_seed)
    hv0 = env.hv
    curve, trans = [0.0], []
    for t in range(env.cfg.max_steps):
        a = policy.act(obs, t)
        nxt, r, done, info = env.step(a)
        if record:
            trans.append({"obs": np.asarray(obs), "action": a, "reward": r,
                          "next_obs": np.asarray(nxt), "done": float(done)})
        obs = nxt
        curve.append(info["hv"] - hv0)
        if done:
            break
    return np.array(curve), trans, [lg.expert for lg in env.logs], env.logs


def collect(env, policy, seeds, repeats, tag=""):
    buf = []
    for rep in range(repeats):
        for i, m in enumerate(seeds):
            _, tr, _, _ = rollout(env, policy, m, rep * 10000 + i, record=True)
            buf.extend(tr)
    print(f"  {tag}: {len(buf)} 条 transition")
    return buf


def train_bandit_online(env, bandit, seeds, repeats):
    """bandit 用自己的 UCB 策略跑训练轨迹，边跑边更新。

    这正是它和 greedy 的唯一区别：greedy 被动拟合 random 的数据，
    bandit 主动决定去看哪些 (状态, action)。
    """
    bandit.explore = True
    for rep in range(repeats):
        for i, m in enumerate(seeds):
            obs = env.reset(m, rep * 10000 + i)
            for t in range(env.cfg.max_steps):
                a = bandit.act(obs, t)
                nxt, r, done, _ = env.step(a)
                bandit.update(obs, a, r)
                obs = nxt
                if done:
                    break
    bandit.explore = False
    return bandit


def train_dqn_online(env, dqn, seeds, repeats, updates_per_epoch=300,
                     eps_start=1.0, eps_end=0.1, eps_frac=0.6):
    """epsilon 在训练的前 60% 线性衰减到 0.1，而不是前 10% 就降到底。

    TA 指出稀疏 HV reward 下探索是关键风险，慢衰减是最便宜的缓解手段。
    """
    buf, losses = [], []
    total = repeats * len(seeds)
    step = 0
    for rep in range(repeats):
        for i, m in enumerate(seeds):
            frac = step / max(total * eps_frac, 1)
            dqn.epsilon = max(eps_end, eps_start + (eps_end - eps_start) * frac)
            _, tr, _, _ = rollout(env, dqn, m, rep * 10000 + i, record=True)
            buf.extend(tr)
            step += 1
        losses += dqn.train_on(buf, n_updates=updates_per_epoch)
        print(f"    epoch {rep+1}/{repeats}  eps={dqn.epsilon:.2f}  "
              f"buffer={len(buf)}  loss={np.mean(losses[-100:]):.4f}")
    dqn.epsilon = 0.0
    return dqn, losses


# --------------------------------------------------------------------------
def _worker_init(cfg_dict, blob, env_kw):
    global _ENV, _POLICIES
    # 并行度来自 Pool，torch 必须单线程：否则 8 个 worker x torch 默认 8 线程
    # = 64 个线程抢 8 个核。RDKit 池不碰 torch 所以之前没暴露，
    # editor_kind="gen" 下每个 worker 都要跑自回归采样。
    import torch
    torch.set_num_threads(1)
    _ENV = MolEnv(OracleBank(), EpisodeConfig(**cfg_dict), **env_kw)
    _POLICIES = pickle.loads(blob)


def _worker_eval(job):
    name, smiles, episode_seed = job
    curve, _, actions, logs = rollout(_ENV, _POLICIES[name], smiles, episode_seed)
    eligible = sum(lg.n_scaffold_ok for lg in logs)
    return {
        "policy": name, "smiles": smiles, "episode_seed": episode_seed,
        "curve": curve.tolist(),
        "final": float(curve[-1]),
        "auc": float(curve.sum()),
        "actions": actions,
        "eligible": eligible,
        "proposed": sum(lg.n_proposed for lg in logs),
        "entered": sum(1 for lg in logs if lg.entered_archive),
        # 关键的混淆控制量：每个合格候选换来多少 HV
        "hv_per_eligible": float(curve[-1]) / max(eligible, 1),
        "eligible_curve": np.cumsum([0] + [lg.n_scaffold_ok for lg in logs]).tolist(),
        # proposal §4.3 的二级指标。StepLog 一直在记这两个数，之前只往上传了
        # n_scaffold_ok，所以 validity 从已有 raw.json 反推不出来。
        "n_valid": sum(lg.n_valid for lg in logs),
        "n_novel": sum(lg.n_novel for lg in logs),
        # uniqueness / novelty / Morgan 内部多样性要的是分子本身，计数反推不出。
        # archive 是 Pareto 前沿，几十个，raw.json 只多几 MB，省掉一次重跑。
        "archive": [s for s, _ in _ENV.archive],
    }


def bootstrap_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--train-repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="训练种子：换它可以量化『重训一次结果会变多少』")
    ap.add_argument("--editor-kind", choices=["rdkit", "gen"], default="rdkit",
                    help="gen = proposal §3.1 的条件生成器主实验")
    ap.add_argument("--gen-ckpt", default="results_gen/generator.pt")
    ap.add_argument("--pool-overlap", type=float, default=0.0,
                    help="rdkit 池的重叠比例：0=按设计独立，1=四个 editor 等价。"
                         "用来扫 action 空间维度（findings §11c）")
    ap.add_argument("--qed-disjoint", action="store_true",
                    help="qed 偏重时用 _QED_SWAP 而非 sa 的 _SHRINK，把维度往上推")
    ap.add_argument("--escape-bits", action="store_true",
                    help="观测里加 4 位『哪些 editor 已在当前分子上失败』")
    args = ap.parse_args()

    if args.quick:
        args.n_train, args.n_test = 40, 20
        args.repeats, args.train_repeats = 1, 2

    data = Path(args.data)
    train_seeds = data.joinpath("seeds_train.txt").read_text().split()[: args.n_train]
    test_seeds = data.joinpath("seeds_test.txt").read_text().split()[: args.n_test]
    cfg_dict = {"max_steps": args.max_steps, "lam": args.lam}
    env_kw = {"use_escape_bits": args.escape_bits,
              "editor_kind": args.editor_kind, "gen_ckpt": args.gen_ckpt,
              "pool_overlap": args.pool_overlap,
              "qed_disjoint": args.qed_disjoint}
    env = MolEnv(OracleBank(), EpisodeConfig(**cfg_dict), **env_kw)

    print(f"train {len(train_seeds)} / test {len(test_seeds)} seeds, "
          f"{args.max_steps} steps, lam={args.lam}, seed={args.seed}, "
          f"escape_bits={args.escape_bits}, editors={args.editor_kind}, "
          f"overlap={args.pool_overlap}, qed_disjoint={args.qed_disjoint}\n")

    policies = make_baselines(seed=args.seed)

    # ---------- 1. random 轨迹：给 greedy 和 gap 用 ----------
    t0 = time.perf_counter()
    print("1. 收集 random 轨迹...")
    buf = collect(env, make_baselines(seed=7 + 13 * args.seed)["random"],
                  train_seeds, args.train_repeats, "random")
    policies["gap"] = GapPolicy().fit(buf)
    policies["greedy"] = GreedyPolicy().fit(buf)
    print(f"   {time.perf_counter() - t0:.0f}s\n")

    # ---------- 2. bandit：在线 UCB ----------
    t0 = time.perf_counter()
    print("2. bandit 在线训练...")
    bandit = LinUCBBandit()
    bandit._init(env.obs_dim)
    policies["bandit"] = train_bandit_online(env, bandit, train_seeds,
                                             args.train_repeats)
    print(f"   {time.perf_counter() - t0:.0f}s\n")

    # ---------- 3. DQN：在线 epsilon-greedy ----------
    t0 = time.perf_counter()
    print("3. DQN 在线训练...")
    dqn, losses = train_dqn_online(env,
                                   DQNPolicy(obs_dim=env.obs_dim, seed=args.seed),
                                   train_seeds, args.train_repeats)
    policies["dqn"] = dqn
    print(f"   {time.perf_counter() - t0:.0f}s  "
          f"loss {np.mean(losses[:100]):.4f} -> {np.mean(losses[-100:]):.4f}\n")

    # ---------- 4. 配对评估 ----------
    jobs = [(name, m, rep * 10000 + i)
            for name in policies
            for rep in range(args.repeats)
            for i, m in enumerate(test_seeds)]
    print(f"4. 评估 {len(jobs)} 个 episode（{args.workers} 进程）...")
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_worker_init,
              initargs=(cfg_dict, pickle.dumps(policies), env_kw)) as pool:
        rows = pool.map(_worker_eval, jobs, chunksize=4)
    dt = time.perf_counter() - t0
    print(f"   {dt/60:.1f} 分钟\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(rows))
    (out / "policies.pkl").write_bytes(pickle.dumps(policies))

    def pick(name, key="final"):
        return {(r["smiles"], r["episode_seed"]): r[key]
                for r in rows if r["policy"] == name}

    # ---------- 5. 汇总 ----------
    print(f"{'policy':13}{'终点HV':>9}{'AUC':>8}{'合格候选':>9}{'HV/候选':>10}  action 分布")
    for name in policies:
        sub = [r for r in rows if r["policy"] == name]
        f = np.mean([r["final"] for r in sub])
        a = np.mean([r["auc"] for r in sub])
        el = np.mean([r["eligible"] for r in sub])
        hpe = np.mean([r["hv_per_eligible"] for r in sub])
        counts = {e: 0 for e in EXPERTS}
        for r in sub:
            for act in r["actions"]:
                counts[act] += 1
        tot = max(sum(counts.values()), 1)
        dist = " ".join(f"{e}:{counts[e]/tot:.0%}" for e in EXPERTS)
        print(f"{name:13}{f:>9.4f}{a:>8.3f}{el:>9.1f}{hpe:>10.5f}  {dist}")

    print("\n『合格候选』列是混淆变量：不同 expert 的产出率差一倍，"
          "\n偏好高产 expert 的 policy 会自动拿到更多候选。"
          "\n『HV/候选』控制了这一项——两列排序不一致时，以后者为准。\n")

    print(f"配对比较 vs random（bootstrap 95% CI，n={len(pick('random'))}）")
    base = pick("random")
    for name in policies:
        if name == "random":
            continue
        cur = pick(name)
        keys = sorted(set(base) & set(cur))
        d = np.array([cur[k] - base[k] for k in keys])
        lo, hi = bootstrap_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {name:13}{d.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}  "
              f"赢 {int((d > 0).sum())}/{len(d)}")

    print("\n核心对比：")
    for a_, b_, why in [("bandit", "greedy", "探索的价值"),
                        ("dqn", "bandit", "延迟 credit assignment 的价值"),
                        ("dqn", "greedy", "两者合计")]:
        pa, pb = pick(a_), pick(b_)
        keys = sorted(set(pa) & set(pb))
        d = np.array([pa[k] - pb[k] for k in keys])
        lo, hi = bootstrap_ci(d)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {a_:7}-{b_:9}{d.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}  "
              f"赢 {int((d > 0).sum())}/{len(d)}   ({why})")

    # ---------- 6. 中途预算点 + 候选口径 ----------
    # HV 曲线在第 5 步左右就饱和，只看终点会把策略差异稀释掉。
    # 这里把同样的配对比较搬到早期预算点，以及换成候选数口径。
    pairs = [(n, "random") for n in policies if n != "random"]
    pairs += [("bandit", "greedy"), ("dqn", "bandit")]

    mid_points = sorted({3, 5, args.max_steps // 2, args.max_steps})
    for x in mid_points:
        res = [summarize_pair(rows, a, b, lambda r: hv_at_step(r, x))
               for a, b in pairs]
        print_pair_table(res, f"配对比较 @ 第 {x} 次 editor 调用")

    # 候选口径：取一个大多数 episode 都能攒够的预算点
    if "eligible_curve" in rows[0]:
        # 公共支撑：只用所有 policy 都攒够 x 个候选的 episode。
        # 按 policy 各自丢弃够不到的 episode 会引入选择偏差——
        # 被丢掉的恰恰是产出率低的轨迹，而产出率正是要控制的那个变量。
        x_cand = pick_candidate_budget(rows, 0.8)
        keys = common_support(rows, x_cand)
        cov = coverage_at(rows, x_cand)
        print(f"\n候选预算点 x = {x_cand}，公共支撑 {len(keys)} 个 episode "
              f"（{cov['_common']:.0%}）")
        res = [summarize_pair(rows, a, b, lambda r: hv_at_eligible(r, x_cand),
                              restrict=keys)
               for a, b in pairs]
        print_pair_table(res, f"配对比较 @ 累计 {x_cand} 个合格候选（控制产出率）")
        print("\n  ↑ 与上面 editor 调用口径的表并列读：两张表结论不一致时，"
              "\n    说明差异来自产出率而非路由质量——这个不一致本身是结果。")

    (out / "summary.json").write_text(json.dumps({
        "n_train": len(train_seeds), "n_test": len(test_seeds),
        "repeats": args.repeats, "train_repeats": args.train_repeats,
        "max_steps": args.max_steps, "lam": args.lam,
        "seed": args.seed, "escape_bits": args.escape_bits,
        "editor_kind": args.editor_kind,
        "eval_minutes": dt / 60,
    }, indent=2))
    print(f"\n写入 {out}/raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())