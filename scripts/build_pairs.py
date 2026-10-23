"""把分子语料变成"编辑对"：同一 Bemis-Murcko 骨架内的类似物对。

为什么是对而不是单分子：editor 的工作是**改**当前分子，不是凭骨架重新造一个。
以当前分子为条件、让模型学 (src, 目标档位) -> dst，模型只要学会"大部分照抄、
按档位改一点"，保 scaffold 就是容易学的行为。scaffold-only 条件下冒烟测出来
保持率是 0%（见 src/generator.py:make_prefix 的注释）。

同骨架类似物对正好就是 lead optimization 里的 analog series，所以这个构造
不是为了迁就模型，它本身就是任务的正确形式化。

两个方向都保留（A->B 和 B->A）：只留"变好"的方向会让条件档位变成冗余信息，
模型会学会无视条件、一律往好里改 —— 那 routing 又没得选了。

  python scripts/build_pairs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interfaces import OBJECTIVES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/gen_corpus.jsonl")
    ap.add_argument("--out", default="data/gen_pairs.jsonl")
    ap.add_argument("--meta-in", default="data/gen_meta.json")
    ap.add_argument("--meta-out", default="data/gen_pairs_meta.json")
    ap.add_argument("--max-per-scaffold", type=int, default=30,
                    help="大骨架族会淹没训练集，设上限")
    ap.add_argument("--max-len", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.corpus).read_text().splitlines()]
    meta = json.loads(Path(args.meta_in).read_text())

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["scaffold"]].append(r)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"{len(rows)} 个分子 -> {len(groups)} 个骨架，"
          f"其中 {len(multi)} 个骨架有 >=2 个成员")

    rng = np.random.default_rng(args.seed)
    pairs = []
    for scaf, members in multi.items():
        n = len(members)
        # 全配对会在大族上爆炸（n^2），按上限随机抽
        budget = min(args.max_per_scaffold, n * (n - 1))
        seen: set[tuple[int, int]] = set()
        tries = 0
        while len(seen) < budget and tries < budget * 10:
            tries += 1
            i, j = rng.integers(0, n, 2)
            if i == j or (i, j) in seen:
                continue
            seen.add((int(i), int(j)))
        for i, j in seen:
            a, b = members[i], members[j]
            if len(a["smiles"]) > args.max_len or len(b["smiles"]) > args.max_len:
                continue
            pairs.append({"src": a["smiles"], "dst": b["smiles"],
                          "scaffold": scaf, "bins": b["bins"],
                          "src_bins": a["bins"],
                          "dy": [float(y - x) for x, y in zip(a["y"], b["y"])],
                          "split": b["split"]})

    rng.shuffle(pairs)
    tr = sum(1 for p in pairs if p["split"] == "train")
    print(f"生成 {len(pairs)} 个编辑对（train {tr} / val {len(pairs)-tr}）")

    # 条件有多少信息量：档位真的变了的比例
    changed = sum(1 for p in pairs if p["bins"] != p["src_bins"])
    print(f"  目标档位与源档位不同的对: {changed}/{len(pairs)} = "
          f"{changed/max(len(pairs),1):.1%}")
    print("  这个比例太低的话，条件就是冗余的，模型会学会无视它。")
    per_obj = {o: sum(1 for p in pairs if p["bins"][j] != p["src_bins"][j])
               for j, o in enumerate(OBJECTIVES)}
    for o, c in per_obj.items():
        print(f"    {o:6} 档位变化 {c/max(len(pairs),1):>6.1%}")

    with Path(args.out).open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    meta.update({"n_pairs": len(pairs), "n_pairs_train": tr,
                 "pair_max_per_scaffold": args.max_per_scaffold,
                 "frac_bins_changed": changed / max(len(pairs), 1),
                 "per_objective_bin_change": per_obj,
                 "format": "(src 分子, 目标档位) -> dst 分子，同 scaffold"})
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    print(f"\n写入 {args.out} 和 {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
