"""生成器质量诊断：改变条件，生成分布真的变了吗？

proposal §5 明确要求的那个检查：
  "Generator-quality diagnostics will test whether objective conditioning
   actually changes the generated property distribution."
以及 §3.1：
  "We will validate whether changing the conditioning actually changes the
   generated property distribution."

这是 router 实验的**前置条件**。如果四个条件产出同一个分布，那 action space
又退化成 1 维，和 RDKit 池 2.05/4 的毛病是一回事（findings §1b），
再比 router 就没有意义。所以这个脚本必须在跑 run_experiment 之前通过。

判据（都在同一批 seed 上配对）：
  1. 自身目标提升 > 其它三个条件在该目标上的提升（specialization，和
     analyze_action_space.py 用的是同一个统计量，可以直接和 2.05/4 对比）；
  2. validity / scaffold retention / uniqueness —— proposal §4.4 的 secondary
     metrics，RDKit 池构造上恒为 100%，生成器不是，必须实测；
  3. novelty：生成物有没有出现在生成器自己的训练语料里（背书 vs 生成）。

  python scripts/diagnose_conditioning.py --n-seeds 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (  # noqa: E402
    bootstrap_ci, objective_scales, participation_ratio,
)
from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.gen_editors import make_gen_editors  # noqa: E402
from src.interfaces import OBJECTIVES  # noqa: E402
from src.oracles import OracleBank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_gen/generator.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_val.txt",
                    help="默认用 val —— test 要留给最终 router 比较")
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bump", type=int, default=1,
                    help="自身目标档位上推几格；>=n_bins 即顶到最高档")
    ap.add_argument("--corpus", default="data/gen_corpus.jsonl")
    ap.add_argument("--pairs", default="data/gen_pairs_delta.jsonl",
                    help="取各目标可达变化的尺度，用于归一 eff dim")
    ap.add_argument("--out", default="results_gen/conditioning.json")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]
    bank = OracleBank()
    editors = make_gen_editors(args.ckpt, bank, bump=args.bump)

    train_smiles = set()
    if Path(args.corpus).exists():
        for line in Path(args.corpus).read_text().splitlines():
            train_smiles.add(json.loads(line)["smiles"])
    print(f"{len(seeds)} 个种子，K={args.k}，bump={args.bump}，"
          f"语料 {len(train_smiles)} 条\n")

    n_obj = len(OBJECTIVES)
    # D[s, e, j] = 条件 e 下，该 seed 上所有合格候选在目标 j 上的最大提升
    D = np.full((len(seeds), n_obj, n_obj), np.nan)
    counts = {e: dict(attempts=0, valid=0, scaf=0, uniq=0, novel=0)
              for e in OBJECTIVES}

    for si, smi in enumerate(seeds):
        base = bank.vector(smi)
        if base is None:
            continue
        for ei, ename in enumerate(OBJECTIVES):
            props = editors[ename].propose(smi, args.k, seed=1000 + si)
            c = counts[ename]
            c["attempts"] += args.k        # 正好 K 次尝试，非法也算
            c["valid"] += len(props)
            seen_ok = []
            for p in props:
                if scaffold_preserved(smi, p):
                    seen_ok.append(p)
            c["scaf"] += len(seen_ok)
            c["uniq"] += len(set(seen_ok))
            c["novel"] += sum(1 for p in set(seen_ok) if p not in train_smiles)
            gains = []
            for p in set(seen_ok):
                v = bank.vector(p)
                if v is not None:
                    gains.append(np.array(v) - np.array(base))
            if gains:
                D[si, ei] = np.max(np.stack(gains), axis=0)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(seeds)}")

    print(f"\n{'='*74}\n1) 生成质量（每次调用 K={args.k} 次尝试，非法也计入）\n{'='*74}")
    print(f"{'条件':8}{'有效率':>9}{'保 scaffold':>12}{'去重后':>9}{'新颖':>8}")
    qual = {}
    for e in OBJECTIVES:
        c = counts[e]
        a = max(c["attempts"], 1)
        qual[e] = {k: c[k] / a for k in ("valid", "scaf", "uniq", "novel")}
        qual[e]["attempts"] = c["attempts"]
        print(f"{e:8}{c['valid']/a:>9.1%}{c['scaf']/a:>12.1%}"
              f"{c['uniq']/a:>9.1%}{c['novel']/a:>8.1%}")
    print("\n  『新颖』= 不在生成器训练语料里。偏低说明模型在背书而不是生成。")

    print(f"\n{'='*74}\n2) 条件真的改变分布了吗（配对，bootstrap 95% CI）\n{'='*74}")
    print("  对手取**均值**而不是最大值：own 是单一条件，max-of-3 即便四个条件")
    print("  完全等价也会系统性胜出（极值的选择偏差），那样的检验永远过不了。")
    print(f"\n{'目标':8}{'自身条件':>11}{'对手均值':>11}{'差':>11}{'95% CI':>24}")
    spec = {}
    n_spec = 0
    for j, o in enumerate(OBJECTIVES):
        own = D[:, j, j]
        rivals = np.delete(D[:, :, j], j, axis=1)
        mean_rival = np.nanmean(rivals, axis=1)
        ok = ~np.isnan(own) & ~np.isnan(mean_rival)
        if ok.sum() < 5:
            print(f"{o:8}  样本不足（n={int(ok.sum())}）")
            continue
        d = own[ok] - mean_rival[ok]
        lo, hi = bootstrap_ci(d)
        sig = lo > 0 or hi < 0
        n_spec += int(sig and d.mean() > 0)
        spec[o] = {"own": float(np.nanmean(own[ok])),
                   "rival": float(np.nanmean(mean_rival[ok])),
                   "mean": float(d.mean()), "lo": lo, "hi": hi,
                   "n": int(ok.sum()),
                   "specialised": bool(sig and d.mean() > 0)}
        star = "*" if sig else " "
        print(f"{o:8}{np.nanmean(own[ok]):>11.4f}"
              f"{np.nanmean(mean_rival[ok]):>11.4f}"
              f"{d.mean():>+11.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")

    # 2b) 去掉"改动幅度"这个混杂：有的条件只是整体改得更狠，于是在所有目标上
    # 都赢。按行归一化后再看对角是否占优，问的才是"条件有没有各管一摊"。
    Mraw = np.nan_to_num(np.nanmean(D, axis=0))
    rown = Mraw / (np.abs(Mraw).sum(axis=1, keepdims=True) + 1e-12)
    diag_wins = [int(rown[:, j].argmax() == j) for j in range(n_obj)]
    print(f"\n  行归一化后（剔除改动幅度的混杂）每个目标上最强的条件：")
    for j, o in enumerate(OBJECTIVES):
        best = OBJECTIVES[int(rown[:, j].argmax())]
        print(f"    {o:6} -> {best:6}{'  (自身)' if diag_wins[j] else ''}")
    print(f"  对角占优 {sum(diag_wins)}/{n_obj}（随机期望 1/4）")

    # 3) 有效 action 维度：和 RDKit 池的 2.05/4 直接可比
    M = np.nanmean(D, axis=0)                      # [editor, objective]
    M = np.nan_to_num(M)
    sv = np.linalg.svd(M, compute_uv=False)
    eff = float(sv.sum() ** 2 / max((sv ** 2).sum(), 1e-12))
    norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = norm @ norm.T

    # 按各目标的可达变化尺度归一后再算 —— 不归一时 dy 尺度最大的目标
    # （qed，std 是 jnk3 的 4.2 倍）会独占奇异值。见 analysis.objective_scales。
    s = objective_scales(args.pairs)
    eff_n = participation_ratio(M / s) if s is not None else float("nan")

    print(f"\n{'='*74}\n3) action space\n{'='*74}")
    print(f"奇异值 {np.round(sv, 4).tolist()}")
    print(f"有效 action 维度（未归一） {eff:.2f} / {n_obj}"
          f"      （RDKit 反应池 2.05）")
    print(f"有效 action 维度（按目标归一）**{eff_n:.2f} / {n_obj}**"
          f"  （RDKit 反应池 2.20，可达上限 3.45）")
    print("\n条件之间的余弦相似度（越接近 1 越说明两个条件是同一个 editor）:")
    print("        " + "".join(f"{o:>9}" for o in OBJECTIVES))
    for i, o in enumerate(OBJECTIVES):
        print(f"{o:8}" + "".join(f"{cos[i, j]:>9.3f}" for j in range(n_obj)))

    verdict = (eff >= 3.0 and n_spec >= 3)
    print(f"\n{'='*74}")
    print(f"结论：{n_spec}/4 个条件在自己的目标上胜出，有效维度 {eff:.2f}/4")
    print("通过 —— 可以拿去跑 router 比较。" if verdict else
          "未通过 —— action space 仍然退化，先修生成器/条件，再谈 routing。")
    print("="*74)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "quality": qual, "specialization": spec, "n_specialized": n_spec,
        "effective_action_dim": eff,
        "effective_action_dim_scaled": eff_n,
        "singular_values": sv.tolist(),
        "editor_cosine": cos.tolist(), "objectives": list(OBJECTIVES),
        "mean_best_delta": M.tolist(), "n_seeds": len(seeds),
        "k": args.k, "bump": args.bump,
        "row_normalized": rown.tolist(), "diag_wins": sum(diag_wins),
        "passes": bool(verdict), "rdkit_baseline_eff_dim": 2.05,
        "rdkit_baseline_eff_dim_scaled": 2.20,
        "achievable_eff_dim_scaled": 3.45,
    }, indent=2))
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
