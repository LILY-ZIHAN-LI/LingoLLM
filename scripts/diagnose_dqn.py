"""DQN 失败诊断：区分"学不出来"和"学到了但序列性无用"。

DQN 在评估时塌缩成常数策略（gsk3b 68%）。在 report 里宣称"序列学习没用"之前，
必须排除实现问题。两种可能的解释有完全不同的 Q 值指纹：

  A. 网络没学到任何区分（实现/优化问题）
     -> 同一状态下各 action 的 Q 几乎相等（相对差 < 1%），
        且 Q 几乎不随状态变化。argmax 由随机的初始化偏置决定。

  B. 网络学到了，但最优策略确实接近常数（科学结论）
     -> 同一状态下 Q 有明显差距，且 Q 随状态变化；
        只是 argmax 恰好大多指向同一个 action。

对照组是 GreedyPolicy：同样的特征、同样的 reward、同样的训练数据，
只是线性模型 + 只看即时收益。如果 greedy 的相对差远大于 DQN，
那 DQN 的问题是优化，不是信号不存在。

用法:
  python scripts/diagnose_dqn.py
  python scripts/diagnose_dqn.py --n-seeds 30 --max-steps 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import MolEnv  # noqa: E402
from src.interfaces import EXPERTS, EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

EPS = 1e-12


# --------------------------------------------------------------------------
def q_values(dqn, obs: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        x = torch.tensor(np.asarray(obs, dtype=np.float32))
        return dqn.q(x).numpy().astype(float)


def greedy_values(greedy, obs: np.ndarray) -> np.ndarray:
    """GreedyPolicy 的预测即时 reward，作为"信号是否存在"的对照。"""
    x = np.append(np.asarray(obs, dtype=float), 1.0)
    return np.asarray(greedy.W @ x, dtype=float)


def collect_states(env, policy, seeds, episode_seed_base=0):
    """沿 policy 自己的轨迹收集访问到的状态。

    用 policy 自己的状态分布，而不是随便造的状态：DQN 只在它实际会遇到的
    状态上塌缩才是问题。同时也收集 random 的状态分布做对照，
    排除"DQN 只是走进了一个各 action 本来就等价的狭窄区域"。
    """
    states = []
    for i, smi in enumerate(seeds):
        obs = env.reset(smi, episode_seed_base + i)
        if hasattr(policy, "reset"):
            policy.reset(episode_seed_base + i)
        for t in range(env.cfg.max_steps):
            states.append(np.asarray(obs, dtype=float))
            a = policy.act(obs, t)
            obs, _, done, _ = env.step(a)
            if done:
                break
    return np.array(states)


# --------------------------------------------------------------------------
def spread_stats(V: np.ndarray) -> dict:
    """V: (n_states, n_actions)。量化"同一状态下各 action 的差异有多大"。

    相对差的分母用 |Q| 的均值（而不是 max-min 的某个尺度），
    因为问题恰恰是"差异相对 Q 的量级可忽略"。
    """
    rng = V.max(axis=1) - V.min(axis=1)          # 每个状态的 action 间极差
    mag = np.abs(V).mean(axis=1)                 # 每个状态的 Q 量级
    rel = rng / np.maximum(mag, EPS)
    argmax = V.argmax(axis=1)
    counts = np.bincount(argmax, minlength=len(EXPERTS))
    return {
        "n_states": int(V.shape[0]),
        "q_mean": float(V.mean()),
        "q_abs_mean": float(np.abs(V).mean()),
        # 各 action 的 Q 在状态间的波动：接近 0 说明网络是个常数函数
        "per_action_std_across_states": V.std(axis=0).round(6).tolist(),
        "per_action_mean": V.mean(axis=0).round(6).tolist(),
        "range_mean": float(rng.mean()),
        "range_median": float(np.median(rng)),
        "rel_range_mean": float(rel.mean()),
        "rel_range_median": float(np.median(rel)),
        "frac_states_rel_below_1pct": float((rel < 0.01).mean()),
        "argmax_counts": counts.tolist(),
        "argmax_frac": (counts / max(V.shape[0], 1)).round(4).tolist(),
        "argmax_entropy_bits": float(
            -sum(p * np.log2(p) for p in counts / max(V.shape[0], 1) if p > 0)
        ),
    }


def advantage_stats(V: np.ndarray) -> dict:
    """把 Q(s,a) 分解成 V(s) + A(s,a)，看网络到底学到了什么。

    第一轮诊断显示 Q 跨状态的 std（约 4.4）远大于同一状态内 action 间的极差
    （约 0.5）。这说明网络的容量几乎全用在拟合"这个状态还能涨多少 HV"（V），
    而不是"该调哪个 editor"（A）。如果 A 还几乎不随状态变化，
    那它等价于一个固定的 action 偏好表——学到的"策略"里没有任何状态依赖，
    这正好解释了评估时的常数塌缩。
    """
    Vs = V.mean(axis=1, keepdims=True)           # V(s)：状态价值
    A = V - Vs                                   # A(s,a)：优势
    total = V.var()
    return {
        "var_total": float(total),
        # Q 的方差里有多少来自"状态之间的差异"，多少来自"同状态内 action 的差异"
        "frac_var_from_state": float(
            np.repeat(Vs, V.shape[1], axis=1).var() / max(total, EPS)),
        "frac_var_from_action": float(A.var() / max(total, EPS)),
        "adv_mean_per_action": A.mean(axis=0).round(6).tolist(),
        # advantage 跨状态的 std vs 它的平均绝对值：
        # 远小于 1 说明 advantage 基本是常数偏置，不随状态变化
        "adv_std_per_action": A.std(axis=0).round(6).tolist(),
        "adv_std_over_abs_mean": float(
            A.std(axis=0).mean() / max(np.abs(A.mean(axis=0)).mean(), EPS)),
    }


def print_advantage(title: str, ad: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  Q 方差分解：状态间 {ad['frac_var_from_state']:.2%}  /  "
          f"同状态 action 间 {ad['frac_var_from_action']:.2%}")
    print("  advantage 均值         " + "  ".join(
        f"{e}:{v:+.4f}" for e, v in zip(EXPERTS, ad["adv_mean_per_action"])))
    print("  advantage 跨状态 std   " + "  ".join(
        f"{e}:{v:.4f}" for e, v in zip(EXPERTS, ad["adv_std_per_action"])))
    print(f"  std / |均值| 比        {ad['adv_std_over_abs_mean']:.2f}"
          "   (<1 = advantage 近似常数偏置；>1 = 有状态依赖)")


def print_block(title: str, st: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  states                 {st['n_states']}")
    print(f"  mean |Q|               {st['q_abs_mean']:.4f}")
    print(f"  action 间极差 (mean)   {st['range_mean']:.6f}")
    print(f"  相对极差 (mean/median) {st['rel_range_mean']:.4%} / "
          f"{st['rel_range_median']:.4%}")
    print(f"  相对极差 <1% 的状态占比 {st['frac_states_rel_below_1pct']:.1%}")
    print("  各 action 均值         " + "  ".join(
        f"{e}:{v:+.4f}" for e, v in zip(EXPERTS, st["per_action_mean"])))
    print("  各 action 跨状态 std   " + "  ".join(
        f"{e}:{v:.5f}" for e, v in zip(EXPERTS, st["per_action_std_across_states"])))
    print("  argmax 分布            " + "  ".join(
        f"{e}:{p:.0%}" for e, p in zip(EXPERTS, st["argmax_frac"]))
          + f"   (entropy {st['argmax_entropy_bits']:.2f}/2.00 bits)")


def sample_table(V: np.ndarray, n: int = 8) -> None:
    print(f"\n  前 {n} 个状态的原始 Q 值：")
    print("    " + "".join(f"{e:>12}" for e in EXPERTS) + "     极差    相对")
    for row in V[:n]:
        rng = row.max() - row.min()
        rel = rng / max(np.abs(row).mean(), EPS)
        print("    " + "".join(f"{v:>12.5f}" for v in row)
              + f"  {rng:>8.5f}  {rel:>6.2%}")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="results/policies.pkl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/dqn_diagnosis.json")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=20)
    args = ap.parse_args()

    policies = pickle.loads(Path(args.policies).read_bytes())
    dqn = policies["dqn"]
    greedy = policies.get("greedy")

    test_seeds = Path(args.data).joinpath("seeds_test.txt").read_text().split()
    test_seeds = test_seeds[: args.n_seeds]

    cfg = EpisodeConfig(max_steps=args.max_steps)
    env = MolEnv(OracleBank(), cfg)

    print(f"DQN 诊断：{len(test_seeds)} 个测试种子，{args.max_steps} 步")
    print(f"reward_scale={dqn.reward_scale}  gamma={dqn.gamma}  "
          f"epsilon={dqn.epsilon}")
    # reward 被放大 reward_scale 倍，所以 Q 的量级应该是
    # (每步 HV 增益 × scale) 在 gamma 折扣下的累加
    print(f"  -> 若每步 HV 增益约 1e-3，Q 的量级应在 "
          f"{1e-3 * dqn.reward_scale / (1 - dqn.gamma):.1f} 附近")

    report = {"config": {"n_seeds": len(test_seeds), "max_steps": args.max_steps,
                         "reward_scale": dqn.reward_scale, "gamma": dqn.gamma}}

    # DQN 自己的状态分布
    print("\n收集状态（DQN 自己的轨迹）...")
    S_dqn = collect_states(env, dqn, test_seeds, 0)
    V = np.array([q_values(dqn, s) for s in S_dqn])
    st = spread_stats(V)
    print_block("Q 值 @ DQN 访问的状态", st)
    sample_table(V)
    report["dqn_on_own_states"] = st

    # random 的状态分布：排除"状态区域太窄"的解释
    print("\n收集状态（random 轨迹，对照状态分布）...")
    S_rnd = collect_states(env, policies["random"], test_seeds, 500000)
    V_rnd = np.array([q_values(dqn, s) for s in S_rnd])
    st_rnd = spread_stats(V_rnd)
    print_block("Q 值 @ random 访问的状态", st_rnd)
    report["dqn_on_random_states"] = st_rnd

    # Q = V(s) + A(s,a) 分解：区分"学到状态价值"和"学到路由"
    ad = advantage_stats(V)
    print_advantage("Q 分解 @ DQN 自己的状态", ad)
    report["advantage_decomposition"] = ad

    # 对照组：同样特征、同样 reward 的线性模型区分度如何
    if greedy is not None and getattr(greedy, "W", None) is not None:
        G = np.array([greedy_values(greedy, s) for s in S_rnd])
        st_g = spread_stats(G)
        print_block("对照：GreedyPolicy 预测即时 reward @ 同一批状态", st_g)
        report["greedy_on_random_states"] = st_g
        ratio = st_g["rel_range_mean"] / max(st_rnd["rel_range_mean"], EPS)
        print(f"\n  greedy 的相对区分度是 DQN 的 {ratio:.0f} 倍")
        report["greedy_over_dqn_rel_spread"] = float(ratio)
        agree = float((G.argmax(axis=1) == V_rnd.argmax(axis=1)).mean())
        print(f"  两者 argmax 一致率 {agree:.1%}（随机基线 25%）")
        report["dqn_greedy_argmax_agreement"] = agree

    # ---------------- 结论 ----------------
    rel = st["rel_range_mean"]
    const_fn = max(st["per_action_std_across_states"]) < 0.01 * st["q_abs_mean"]
    state_dep_adv = ad["adv_std_over_abs_mean"] > 1.0
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    if rel < 0.01 or const_fn:
        verdict = "A: 网络没学到任何 action 区分（实现/优化失败，非科学结论）"
        print(f"  action 间相对差 {rel:.4%} < 1%：网络对所有 action 给出几乎相同的 Q。")
        print("  argmax 由初始化噪声决定，不携带信息。")
        print("  -> 不能据此宣称『序列学习没用』，只能说『在本预算下 DQN 学不出来』。")
    elif not state_dep_adv:
        verdict = ("B1: 学到了 V(s) 但 advantage 近似常数偏置 —— "
                   "拟合成功，策略无状态依赖")
        print(f"  action 间相对差 {rel:.4%}：Q 确实区分 action，排除解释 A。")
        print(f"  但 Q 的方差 {ad['frac_var_from_state']:.0%} 来自状态间、"
              f"仅 {ad['frac_var_from_action']:.0%} 来自同状态的 action 间。")
        print(f"  advantage 跨状态 std / |均值| = {ad['adv_std_over_abs_mean']:.2f} < 1，"
              "即 advantage 基本是固定偏好表。")
        print("  -> 网络把容量用在预测『这个状态还能涨多少 HV』（价值），"
              "\n     而没能学出『该调哪个 editor』（路由）。塌缩是这个分解的直接后果。")
        print("  -> 可以写成『Q 拟合成功但路由信号极弱』，比『学不出来』更强，"
              "\n     但仍不等于『序列性原则上无用』。")
    else:
        verdict = ("B2: 学到了状态依赖的路由；排除实现失败，"
                   "路由信号真实但量级极小")
        print(f"  action 间相对差 {rel:.4%} >= 1%，且 advantage 随状态变化"
              f"（std/|均值| = {ad['adv_std_over_abs_mean']:.2f} > 1）。")
        print(f"  argmax 分布的熵 {st['argmax_entropy_bits']:.2f} bits（纯常数为 0），"
              "所以严格说不是常数策略。")
        print(f"  但 action 相关的方差只占 Q 总方差的 "
              f"{ad['frac_var_from_action']:.2%}：")
        print("  -> 结论是『实现没问题，路由信号相对价值信号小两个数量级』，"
              "\n     而不是『DQN 学不出来』。这与 greedy/bandit 在 editor 调用"
              "\n     口径下也只有微弱优势是一致的。")
    report["verdict"] = verdict
    report["constant_function"] = bool(const_fn)
    report["state_dependent_advantage"] = bool(state_dep_adv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
