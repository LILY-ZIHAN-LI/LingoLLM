"""生成式四个专家到底有多像？—— 决定 final 的专家专业化要做到哪一档。

背景：findings §11 测到生成式支的有效 action 维度只有 1.80，§12 的同族曲线
说这个维度解释了它丢掉 routing 收益。但 1.80 这个数是从**下游增益矩阵**
（editor × objective 的平均最好提升）算出来的，它分不开两种完全不同的病因：

  (i) 条件根本没进模型 —— 换 bin token 不改变输出分布；
 (ii) 条件进了模型，但解码（temperature 0.9 / top-k 24）或者 bump=1 太弱，
      把差异洗掉了。

两者对应的修法差一个量级：(ii) 只要调解码或 bump，(i) 必须动权重
（adapter 或者带互斥项重训）。所以这里直接量两件事：

  A) logit 空间的条件敏感度。同一个母分子、teacher-force 它自己的 token 串，
     只换前缀里的 bin token，比较每一步的 next-token 分布。
     这是**纯前向**，不采样、不碰 RDKit，所以快，而且把条件的作用完全隔离出来
     （上下文逐字相同，唯一的差别就是那 4 个 bin token）。
     用全距距离（TV ∈ [0,1]）：0 = 分布完全相同，1 = 支撑不相交。

     bump 不是一个可扫的轴：delta 口径的中性档是 2、n_bins=4，
     min(2+bump, 3) 在 bump>=1 时一律顶到第 3 档，所以 bump=1 已经是顶格，
     gen_editors.py 注释里怀疑的"bump=1 太弱"在构造上不成立。
     能拉开的最大条件对比是**同一维取第 0 档 vs 第 3 档**，这里量的是它。

     阳性对照是必须的：TV 很小可能是条件没用，也可能是我这个量法本身
     看不见任何东西。所以额外量"把前缀里的源分子换成另一个分子"——
     上下文语义被彻底改掉，TV 必须大。它不大，说明测量无效，而不是条件无效。

  B) 端到端的提案集重叠。同一个 (分子, seed) 上真的调 propose()，
     用跑实验时那套解码参数，算四个专家提案集合的两两 Jaccard。
     A 大而 B 也大 => 病因是解码；A 就小 => 病因是条件本身。

  python scripts/diagnose_expert_overlap.py --n-seeds 60
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import canonical                              # noqa: E402
from src.gen_editors import TEMPERATURE, TOP_K              # noqa: E402
from src.gen_editors import load_backbone, make_gen_editors  # noqa: E402
from src.generator import make_prefix, tokenize             # noqa: E402
from src.interfaces import OBJECTIVES                       # noqa: E402
from src.oracles import OracleBank                          # noqa: E402

PAIRS = list(itertools.combinations(range(len(OBJECTIVES)), 2))


def tv_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐位置的全距距离。a, b 是 [T, V] 的概率矩阵。"""
    return 0.5 * np.abs(a - b).sum(axis=1)


def dists(model, prefix: list[int], cont: list[int]) -> np.ndarray | None:
    """teacher-force cont，返回每一步的 next-token 分布 [T, V]。

    位置对齐：模型在位置 i 的输出预测 token i+1，所以取
    [len(prefix)-1, len(prefix)+len(cont)-2] 这段，正好是"下一个 token 属于
    分子本身"的全部位置。
    """
    ids = prefix + cont
    if len(ids) > model.cfg.max_len:
        return None
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))[0]
    lo, hi = len(prefix) - 1, len(prefix) + len(cont) - 1
    return torch.softmax(logits[lo:hi].float(), dim=-1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_gen/generator_delta.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds-file", default="seeds_val.txt")
    ap.add_argument("--n-seeds", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="results_gen/expert_overlap.json")
    args = ap.parse_args()

    seeds = Path(args.data).joinpath(args.seeds_file).read_text().split()
    seeds = seeds[: args.n_seeds]

    model, vocab, meta = load_backbone(args.ckpt)
    n_bins = meta["n_bins"]
    mode = meta.get("conditioning", "absolute")
    if mode != "delta":
        print(f"!! 这个 ckpt 是 {mode} 口径；A 部分的中性参照只对 delta 有定义")
    neutral = [meta["neutral_bin"][o] for o in OBJECTIVES]
    top = n_bins - 1
    saturated = all(min(v + 1, top) == top for v in neutral)
    print(f"ckpt={args.ckpt}  模式={mode}  n_bins={n_bins}  "
          f"中性档={neutral}  顶档={top}")
    print(f"bump 是否在 bump=1 就顶格：{saturated}"
          f"{'  => bump 不是可扫的轴' if saturated else ''}")
    print(f"{len(seeds)} 个母分子\n")

    # ---- A) logit 空间 -------------------------------------------------
    acc = {"pair": [], "pair_max": [], "vs_neutral": [[] for _ in OBJECTIVES],
           "min_vs_max": [[] for _ in OBJECTIVES], "control": []}
    n_used = 0
    for si, smi in enumerate(seeds):
        parent = canonical(smi)
        if parent is None:
            continue
        src = tokenize(parent, vocab)
        cont = src + [vocab["<eos>"]]
        base = dists(model, make_prefix(neutral, src, vocab), cont)
        if base is None:
            continue

        P = []            # 四个专家实际用的条件：自己那维顶格，其余中性
        ok = True
        for j in range(len(OBJECTIVES)):
            tgt = list(neutral)
            tgt[j] = top
            d = dists(model, make_prefix(tgt, src, vocab), cont)
            # 同一维取最低档：这是能拉开的最大条件对比
            lo_t = list(neutral)
            lo_t[j] = 0
            dlo = dists(model, make_prefix(lo_t, src, vocab), cont)
            if d is None or dlo is None:
                ok = False
                break
            P.append(d)
            acc["vs_neutral"][j].append(float(tv_rows(d, base).mean()))
            acc["min_vs_max"][j].append(float(tv_rows(d, dlo).mean()))
        if not ok:
            continue

        # 阳性对照：条件 bin 不变，只把前缀里的源分子换成另一个分子。
        # 续写串仍是本分子，所以位置照旧对齐。
        alt = canonical(seeds[(si + 1) % len(seeds)])
        dctl = (dists(model, make_prefix(neutral, tokenize(alt, vocab), vocab),
                      cont) if alt and alt != parent else None)
        if dctl is not None:
            acc["control"].append(float(tv_rows(dctl, base).mean()))

        pv = [tv_rows(P[i], P[j]) for i, j in PAIRS]
        acc["pair"].append(float(np.mean([x.mean() for x in pv])))
        acc["pair_max"].append(float(np.max([x.max() for x in pv])))
        n_used += 1

    pair = float(np.mean(acc["pair"]))
    pmax = float(np.mean(acc["pair_max"]))
    ctl = float(np.mean(acc["control"])) if acc["control"] else float("nan")
    vn = [float(np.mean(v)) for v in acc["vs_neutral"]]
    mm = [float(np.mean(v)) for v in acc["min_vs_max"]]
    secA = {"pair_tv": pair, "pair_tv_max": pmax, "vs_neutral": vn,
            "min_vs_max": mm, "positive_control_tv": ctl,
            "n_control": len(acc["control"])}

    print(f"{'='*72}\nA) 条件敏感度（logit 空间，TV ∈ [0,1]，{n_used} 个母分子）"
          f"\n{'='*72}")
    print(f"  四个专家两两 TV（6 个配对、全部解码位置的均值） {pair:>8.4f}")
    print(f"  其中最大单步 TV                               {pmax:>8.4f}")
    print(f"  阳性对照：前缀里换掉源分子                     {ctl:>8.4f}"
          f"   <- 这个必须大")
    print(f"\n  {'目标':8}{'顶格 vs 中性':>14}{'顶格 vs 最低档':>16}")
    for j, o in enumerate(OBJECTIVES):
        print(f"  {o:8}{vn[j]:>14.4f}{mm[j]:>16.4f}")
    print("\n  上下文逐字相同，唯一差别是 4 个 bin token —— 所以这些数就是")
    print("  '换条件改变了多少输出分布'，不含解码随机性。")

    # ---- B) 端到端提案集重叠 -------------------------------------------
    print(f"\n{'='*72}\nB) 提案集重叠（真解码，temperature={TEMPERATURE} "
          f"top_k={TOP_K} K={args.k}）\n{'='*72}")
    # 低 Jaccard 本身**不能**读成"专家各有专长"：四次独立采样同一个分布
    # 也会得到低 Jaccard。所以必须有零假设对照 ——
    # 同一个专家、换一个采样种子，两次提案集的 Jaccard。
    # 专家之间的 Jaccard 若与它持平，说明专家身份除了换种子之外什么也没做。
    bank = OracleBank()
    eds = make_gen_editors(args.ckpt, bank, bump=1)
    jac, only_one, n_prop, same_ed = [], [], [], []
    for si, smi in enumerate(seeds):
        sets = [set(eds[o].propose(smi, args.k, seed=1000 + si))
                for o in OBJECTIVES]
        n_prop.append(float(np.mean([len(s) for s in sets])))
        for i, j in PAIRS:
            u = sets[i] | sets[j]
            if u:
                jac.append(len(sets[i] & sets[j]) / len(u))
        for o, s0 in zip(OBJECTIVES, sets):
            s1 = set(eds[o].propose(smi, args.k, seed=7000 + si))
            u = s0 | s1
            if u:
                same_ed.append(len(s0 & s1) / len(u))
        allm = set().union(*sets)
        if allm:
            cnt = {m: sum(m in s for s in sets) for m in allm}
            only_one.append(sum(1 for v in cnt.values() if v == 1) / len(allm))
    secB = {"jaccard": float(np.mean(jac)) if jac else None,
            "jaccard_same_editor_other_seed": float(np.mean(same_ed)),
            "frac_unique_to_one": float(np.mean(only_one)),
            "mean_proposals_per_call": float(np.mean(n_prop)),
            "bump": 1}
    print(f"  专家之间 两两 Jaccard                 {secB['jaccard']:.3f}")
    print(f"  零假设对照：同一专家换种子 Jaccard     "
          f"{secB['jaccard_same_editor_other_seed']:.3f}   <- 对照")
    print(f"  只被一个专家提出的分子占比             "
          f"{secB['frac_unique_to_one']:.3f}")
    print(f"  每次调用有效提案                      "
          f"{secB['mean_proposals_per_call']:.2f}/{args.k}")

    # ---- 判读 ----------------------------------------------------------
    print(f"\n{'='*72}\n判读\n{'='*72}")
    worst = max(mm)
    jb, jc = secB["jaccard"], secB["jaccard_same_editor_other_seed"]
    if not (ctl > 0.05):
        verdict = (f"测量无效：阳性对照只有 {ctl:.4f}，换掉源分子都没改变分布，"
                   f"所以 TV 小不能解读为条件无效。先查 teacher-force 的位置对齐。")
    elif jb <= jc * 1.3 and worst < 0.1 * ctl:
        verdict = (
            f"专家身份 ≈ 只是换了个采样种子。专家之间的 Jaccard {jb:.3f} 并不"
            f"高于'同一专家换种子'的 {jc:.3f}，而条件的最大对比 {worst:.4f} 只有"
            f"阳性对照 {ctl:.4f} 的 1/{ctl/max(worst,1e-9):.0f}。"
            f"所以 1.80 维的原因是条件几乎没改变分布，低 Jaccard 全部来自采样"
            f"噪声而不是专长。便宜修法（bump 已顶格 / logit bias / 降温）都治不了"
            f"这个，必须动权重：per-objective adapter，或训练时加跨专家互斥项。")
    elif worst < 0.1 * ctl:
        verdict = (f"条件对分布的作用很弱（最大对比 {worst:.4f} vs 对照 "
                   f"{ctl:.4f}），但专家之间的 Jaccard {jb:.3f} 低于同专家换种子的"
                   f" {jc:.3f}，两者不一致，需要先查 B 的实现。")
    elif jb > jc * 1.3:
        verdict = (f"条件有实质作用（最大对比 {worst:.4f}）但提案集比随机重采"
                   f"更像（{jb:.3f} > {jc:.3f}）=> 瓶颈在解码，先降温/收紧 top-k。")
    else:
        verdict = (f"条件有实质作用（最大对比 {worst:.4f}）且专家之间确实比"
                   f"同专家换种子更分散（{jb:.3f} < {jc:.3f}）。"
                   f"那么 1.80 维是'提案不一样但属性效果一样'，要动训练目标。")
    print("  " + verdict)
    print(f"  参照：findings §11 的生成式有效维度 1.80，RDKit 2.20，"
          f"可达上界 3.45。")

    out = {"ckpt": args.ckpt, "n_parents": n_used, "n_seeds": len(seeds),
           "k": args.k, "temperature": TEMPERATURE, "top_k": TOP_K,
           "objectives": list(OBJECTIVES), "neutral_bin": neutral,
           "bump_saturated_at_1": saturated,
           "logit_sensitivity": secA, "proposal_overlap": secB,
           "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
