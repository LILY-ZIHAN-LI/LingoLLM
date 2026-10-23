"""失败分析：局部有益的编辑如何损害后续进展。

proposal 第 4 节承诺要"inspect trajectories where a locally useful edit damages
later progress"。诊断（analyze_editor_depletion.py）已经给出了机制：
某步产出 0 个合格候选 -> 分子不动 -> 确定性 policy 重选同一个 editor -> 又是 0。
这个脚本把机制落到**具体分子**上：

  1. 从 raw.json 里筛"早期领先、终点落后"的 episode（greedy 在第 3 步赢 random，
     但终点输给它）——这正是"短视收益换来长期停滞"的操作化定义。
  2. 对排名最前的几个重放两条轨迹，逐步打印 editor / 当前分子 / 合格候选数 / HV。
  3. 画出分子结构的变化（英文标签，直接进 report）。

  python scripts/analyze_failures.py --top 3
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
from src.interfaces import EpisodeConfig  # noqa: E402
from src.oracles import OracleBank  # noqa: E402

LABEL = {"gsk3b": "GSK3B", "jnk3": "JNK3", "qed": "QED", "sa": "SA"}


def replay(env, policy, smiles: str, episode_seed: int):
    """重放一条轨迹，逐步记录。env 对所有 policy 完全相同，所以可复现。"""
    obs = env.reset(smiles, episode_seed)
    if hasattr(policy, "reset"):
        policy.reset(episode_seed)
    hv0 = env.hv
    steps = []
    for t in range(env.cfg.max_steps):
        a = policy.act(obs, t)
        obs, _, done, info = env.step(a)
        lg = env.logs[-1]
        steps.append({
            "t": t, "expert": lg.expert, "parent": lg.current,
            "n_eligible": lg.n_scaffold_ok, "hv_gain": info["hv"] - hv0,
            "entered": lg.entered_archive,
            "best_single_gain": lg.best_single_gain,
            "next": env.current,
            "moved": env.current != lg.current,
        })
        if done:
            break
    return steps


def print_side_by_side(sa, sb, na, nb) -> None:
    print(f"\n  {'t':>2}  {na:<34}  {nb:<34}")
    print(f"  {'':>2}  {'editor  cand  HV      动':<34}  "
          f"{'editor  cand  HV      动':<34}")
    for t in range(max(len(sa), len(sb))):
        def cell(s):
            if t >= len(s):
                return " " * 34
            r = s[t]
            mark = "→" if r["moved"] else "×"
            return (f"{LABEL[r['expert']]:<7}{r['n_eligible']:>4}  "
                    f"{r['hv_gain']:>7.4f}  {mark}").ljust(34)
        print(f"  {t:>2}  {cell(sa)}  {cell(sb)}")
    print(f"  {'':>2}  × = 该步 0 个合格候选，分子原地不动（吸收态）")


def draw(steps, path: Path, title: str) -> bool:
    """画出轨迹里分子真正发生变化的那几步。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception as exc:  # pragma: no cover
        print(f"  (跳过作图: {exc})")
        return False
    mols, legends = [], []
    first = steps[0]["parent"]
    m = Chem.MolFromSmiles(first)
    if m is not None:
        mols.append(m)
        legends.append("seed")
    for r in steps:
        if not r["moved"]:
            continue
        m = Chem.MolFromSmiles(r["next"])
        if m is None:
            continue
        mols.append(m)
        legends.append(f"t={r['t']} via {LABEL[r['expert']]}\nHV +{r['hv_gain']:.4f}")
    if len(mols) < 2:
        return False
    img = Draw.MolsToGridImage(mols[:8], molsPerRow=min(4, len(mols)),
                               subImgSize=(260, 220), legends=legends[:8])
    # 返回类型随 RDKit 版本和运行环境而变：PIL Image / 带 .data 的包装 / 原始 bytes
    if hasattr(img, "save"):
        img.save(str(path))
    else:
        data = img.data if hasattr(img, "data") else img
        path.write_bytes(data.encode() if isinstance(data, str) else data)
    print(f"  结构图 -> {path}  ({title})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.json")
    ap.add_argument("--policies", default="results/policies.pkl")
    ap.add_argument("--a", default="greedy", help="早期领先的那个 policy")
    ap.add_argument("--b", default="random", help="终点反超的那个 policy")
    ap.add_argument("--early", type=int, default=3, help="早期预算点")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--out", default="results/failure_cases.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.raw).read_text())
    A = {(r["smiles"], r["episode_seed"]): r for r in rows if r["policy"] == args.a}
    B = {(r["smiles"], r["episode_seed"]): r for r in rows if r["policy"] == args.b}
    keys = sorted(set(A) & set(B))

    def early(r):
        return r["curve"][min(args.early, len(r["curve"]) - 1)]

    # ---------- 总体有多普遍 ----------
    lead = [k for k in keys if early(A[k]) > early(B[k])]
    lost = [k for k in lead if A[k]["final"] < B[k]["final"]]
    print(f"{args.a} vs {args.b}：共 {len(keys)} 个配对 episode")
    print(f"  第 {args.early} 步时 {args.a} 领先的： {len(lead)} "
          f"（{len(lead)/max(len(keys),1):.0%}）")
    print(f"  其中到终点反被超过的： {len(lost)} "
          f"（占领先者的 {len(lost)/max(len(lead),1):.0%}）")

    if lost:
        dead_a = np.mean([
            float(np.mean(np.diff(A[k]["eligible_curve"]) == 0)) for k in lost])
        dead_b = np.mean([
            float(np.mean(np.diff(B[k]["eligible_curve"]) == 0)) for k in lost])
        print(f"\n  这些 episode 里的死步占比：{args.a} {dead_a:.0%}  "
              f"vs  {args.b} {dead_b:.0%}")
        print("  -> 早期领先被吐回去，伴随的是大量零产出的空转步。")

    # ---------- 挑最典型的几个重放 ----------
    scored = sorted(lost, key=lambda k: (early(A[k]) - early(B[k]))
                    + (B[k]["final"] - A[k]["final"]), reverse=True)
    picks = scored[: args.top]
    if not picks:
        print("\n没有符合条件的 episode。")
        return 0

    policies = pickle.loads(Path(args.policies).read_bytes())
    env = MolEnv(OracleBank(), EpisodeConfig(max_steps=args.max_steps))
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    cases = []
    for n, (smi, eseed) in enumerate(picks, 1):
        print(f"\n{'='*78}\n案例 {n}: {smi}  (episode_seed={eseed})")
        print(f"  第 {args.early} 步： {args.a} {early(A[(smi,eseed)]):.4f}  vs  "
              f"{args.b} {early(B[(smi,eseed)]):.4f}")
        print(f"  终点：   {args.a} {A[(smi,eseed)]['final']:.4f}  vs  "
              f"{args.b} {B[(smi,eseed)]['final']:.4f}")
        sa = replay(env, policies[args.a], smi, eseed)
        sb = replay(env, policies[args.b], smi, eseed)
        print_side_by_side(sa, sb, args.a, args.b)

        n_dead_a = sum(1 for r in sa if not r["moved"])
        stuck_from = next((r["t"] for r in sa
                           if all(not q["moved"] for q in sa[r["t"]:])), None)
        print(f"\n  {args.a}: {n_dead_a}/{len(sa)} 步没动"
              + (f"，从 t={stuck_from} 起再也没动过" if stuck_from is not None else ""))
        print(f"  {args.b}: {sum(1 for r in sb if not r['moved'])}/{len(sb)} 步没动")

        draw(sa, figdir / f"fig6_case{n}_{args.a}.png", f"case {n} / {args.a}")
        draw(sb, figdir / f"fig6_case{n}_{args.b}.png", f"case {n} / {args.b}")
        cases.append({
            "smiles": smi, "episode_seed": eseed,
            f"early_{args.a}": early(A[(smi, eseed)]),
            f"early_{args.b}": early(B[(smi, eseed)]),
            f"final_{args.a}": A[(smi, eseed)]["final"],
            f"final_{args.b}": B[(smi, eseed)]["final"],
            "stuck_from": stuck_from,
            f"{args.a}_trace": [{k: v for k, v in r.items()} for r in sa],
            f"{args.b}_trace": [{k: v for k, v in r.items()} for r in sb],
        })

    Path(args.out).write_text(json.dumps({
        "a": args.a, "b": args.b, "early_step": args.early,
        "n_pairs": len(keys), "n_lead": len(lead), "n_lead_then_lost": len(lost),
        "cases": cases,
    }, indent=2))
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
