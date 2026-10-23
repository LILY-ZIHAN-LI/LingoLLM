"""把条件从"目标档位"改成"变化分位"。

为什么必须换（证据见 docs/findings.md §10）：
档位边界是在 ChEMBL 百分位上拟合的，而 ChEMBL 对 GSK3B/JNK3 绝大多数无活性。
种子分子是做 lead optimization 的起点、本身偏活性，于是在 40 个 val 种子上
  gsk3b 100% 已在最高档、jnk3 65% 已在最高档。
`min(cur + bump, n_bins - 1)` 等于当前档位 —— gsk3b 这个 editor 的条件和
"保持原样"逐字相同，构造上就是个无条件类似物生成器，它的自身增益因此是负的
（种子在顶格，任何改动只能向语料均值回归）。
"可上推空间"的排序 qed 100% > sa 70% > jnk3 35% > gsk3b 0% 与 gate 里
专精程度的排序完全一致，四个目标无一例外。

换成变化分位之后，`dy = 0` 永远落在中间档而不是顶端，所以**可上推空间恒为
100%**，与分子落在属性空间的哪个位置无关。这也顺带消掉了一整类 bug：新条件
不再依赖当前分子的 oracle 分数，不存在训练/推理分箱口径不一致的可能
（之前那处 `>` vs `>=` 影响了 70.7% 的分子）。

刻意保持 4 档 x 4 目标的 token 布局不变：词表、src/generator.py:make_prefix、
GenEditor 的调用路径都不用改，只换档位的**含义**。

边界只在 train 片段上拟合，与 scripts/prepare_generator_data.py 的口径一致。

  python scripts/rebin_delta.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interfaces import OBJECTIVES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/gen_pairs.jsonl")
    ap.add_argument("--meta-in", default="data/gen_pairs_meta.json")
    ap.add_argument("--out", default="data/gen_pairs_delta.jsonl")
    ap.add_argument("--meta-out", default="data/gen_pairs_delta_meta.json")
    ap.add_argument("--n-bins", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.pairs).read_text().splitlines()]
    meta = json.loads(Path(args.meta_in).read_text())
    print(f"{len(rows)} 个编辑对，切 {args.n_bins} 档变化分位")

    # 只在 train 上拟合边界
    DY_tr = np.array([r["dy"] for r in rows if r["split"] == "train"])
    qs = np.linspace(0, 100, args.n_bins + 1)[1:-1]
    edges = {o: np.percentile(DY_tr[:, j], qs).tolist()
             for j, o in enumerate(OBJECTIVES)}

    def to_bins(dy):
        return [int(np.searchsorted(edges[o], dy[j], side="right"))
                for j, o in enumerate(OBJECTIVES)]

    # dy=0 落在哪一档 —— 推理时"其它目标保持不变"就用这一档
    neutral = {o: int(np.searchsorted(edges[o], 0.0, side="right"))
               for o in OBJECTIVES}

    print(f"\n{'目标':8}{'变化分位边界':>34}{'dy=0 落在':>11}{'顶档可用':>10}")
    for o in OBJECTIVES:
        e = "[" + ", ".join(f"{v:+.4f}" for v in edges[o]) + "]"
        top_ok = neutral[o] < args.n_bins - 1
        print(f"{o:8}{e:>34}{'bin' + str(neutral[o]):>11}"
              f"{'是' if top_ok else '否 <- 仍然顶格!':>10}")
    assert all(neutral[o] < args.n_bins - 1 for o in OBJECTIVES), \
        "dy=0 落在最高档，条件仍然无法要求提升 —— 分档方式还得改"

    counts = np.zeros((len(OBJECTIVES), args.n_bins), int)
    with Path(args.out).open("w") as fh:
        for r in rows:
            b = to_bins(r["dy"])
            for j, v in enumerate(b):
                counts[j, v] += 1
            r["bins_abs"] = r["bins"]          # 留着，便于两种条件对照
            r["bins"] = b
            fh.write(json.dumps(r) + "\n")

    print(f"\n各档占比（应接近 {1/args.n_bins:.0%}）")
    print(f"{'目标':8}" + "".join(f"{'bin'+str(v):>9}" for v in
                                  range(args.n_bins)))
    n = len(rows)
    for j, o in enumerate(OBJECTIVES):
        print(f"{o:8}" + "".join(f"{counts[j, v]/n:>9.1%}"
                                 for v in range(args.n_bins)))

    meta.update({
        "conditioning": "delta",          # src/gen_editors.py 据此选条件方式
        "delta_edges": edges,
        "neutral_bin": neutral,
        "n_bins": args.n_bins,
        "format": "(src 分子, 各目标的变化分位) -> dst 分子，同 scaffold",
    })
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    print(f"\n写入 {args.out} 和 {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
