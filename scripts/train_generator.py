"""训练条件 SMILES transformer（proposal §3.1 的 compact generator）。

  micromamba activate lingollm
  python -u scripts/train_generator.py --epochs 12

loss 只加在 <mol> 之后的 token 上：前缀（属性档位 + scaffold）是给定的条件，
不是要建模的分布。把 loss 加在前缀上会让模型把容量花在背 scaffold 分布上。

CPU-only，所以默认用 6 层 x 256 维（~5M 参数）。torch 线程数按核数设。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chem import canonical, scaffold_preserved  # noqa: E402
from src.generator import (  # noqa: E402
    CondSmilesTransformer, GenConfig, build_vocab, count_params, detokenize,
    make_prefix, split_smiles, tokenize,
)


class ReconDataset(Dataset):
    """一条样本 = 前缀（目标档位 + 源分子）+ 目标分子 SMILES。

    源分子作条件而非 scaffold —— 理由见 src/generator.py:make_prefix。
    """

    def __init__(self, rows, vocab, max_len):
        self.rows, self.vocab, self.max_len = rows, vocab, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        pre = make_prefix(r["bins"], tokenize(r["src"], self.vocab),
                          self.vocab)
        body = tokenize(r["dst"], self.vocab) + [self.vocab["<eos>"]]
        ids = (pre + body)[: self.max_len]
        # loss mask 标在"要被预测的那个位置"上。run_epoch 里取 M[:, 1:]
        # 与 tgt = X[:, 1:] 对齐，所以这里直接按 ids 的下标标：
        # 前缀本身不计 loss，body（含 <eos>）全计。
        mask = [0] * len(pre) + [1] * (len(ids) - len(pre))
        return torch.tensor(ids), torch.tensor(mask[: len(ids)])


class LengthBucketSampler(torch.utils.data.Sampler):
    """按长度分桶出 batch。

    CPU 上这是最大的一笔：attention 是 O(T^2)，而随机成批时 batch 内最长的
    序列（~230）远大于中位数（~110），绝大部分算力花在 pad 上。
    做法是先打乱、切成大块、块内按长度排序再切 batch、最后打乱 batch 顺序 ——
    既保住了随机性，又让同一个 batch 里的序列长度接近。
    """

    def __init__(self, lengths, batch_size: int, chunk: int = 50, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.bs, self.chunk, self.seed, self.epoch = batch_size, chunk, seed, 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths))
        big = self.bs * self.chunk
        batches = []
        for s in range(0, len(idx), big):
            part = idx[s: s + big]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[i: i + self.bs].tolist()
                        for i in range(0, len(part), self.bs)]
        rng.shuffle(batches)
        return iter(batches)


def collate(batch, pad: int):
    L = max(len(x) for x, _ in batch)
    X = torch.full((len(batch), L), pad, dtype=torch.long)
    M = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, m) in enumerate(batch):
        X[i, : len(x)] = x
        M[i, : len(m)] = m
    return X, M


def run_epoch(model, loader, pad, opt=None, sched=None, log_every=200):
    train = opt is not None
    model.train(train)
    tot_loss = tot_tok = 0
    t0 = time.time()
    for it, (X, M) in enumerate(loader):
        inp, tgt = X[:, :-1], X[:, 1:]
        msk = M[:, 1:].bool()                     # 对齐到被预测的位置
        with torch.set_grad_enabled(train):
            logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                reduction="none").reshape(tgt.shape)
            n = msk.sum()
            loss = (loss * msk).sum() / n.clamp(min=1)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        tot_loss += float(loss.detach()) * int(n)
        tot_tok += int(n)
        if train and it % log_every == 0 and it:
            el = time.time() - t0
            print(f"    it {it:5d}  loss {tot_loss/tot_tok:.4f}  "
                  f"{tot_tok/el:7.0f} tok/s  lr {opt.param_groups[0]['lr']:.2e}",
                  flush=True)
    return tot_loss / max(tot_tok, 1)


@torch.no_grad()
def sample_quality(model, rows, vocab, n: int, temperature=0.7, top_k=16):
    """每个 epoch 实采一批，看合法率/保 scaffold 率。

    token CE 不是我们要的指标：CE 0.35 在 60 token 的 SMILES 上，整条正确的
    概率约 e^-21。能不能产出**合法且保骨架**的分子才是 editor 能否用的判据，
    所以直接测它，而不是等训练完再发现不行。
    """
    if not rows or n <= 0:
        return {}
    model.eval()
    inv = {i: tk for tk, i in vocab.items()}
    eos = vocab["<eos>"]
    val = scaf = exact = 0
    g = torch.Generator()
    for i, r in enumerate(rows[:n]):
        pre = make_prefix(r["bins"], tokenize(r["src"], vocab), vocab)
        if len(pre) >= model.cfg.max_len - 4:
            continue
        g.manual_seed(i)
        ids = model.sample(pre, max_new=min(140, model.cfg.max_len - len(pre)),
                           eos=eos, temperature=temperature, top_k=top_k,
                           generator=g)
        cs = canonical(detokenize(ids, inv))
        if cs is None:
            continue
        val += 1
        if scaffold_preserved(r["src"], cs):
            scaf += 1
        if cs == canonical(r["dst"]):
            exact += 1
    m = min(n, len(rows))
    return {"valid": val / m, "scaffold": scaf / m, "exact": exact / m}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/gen_pairs_delta.jsonl")
    ap.add_argument("--gen-meta", default="data/gen_pairs_delta_meta.json")
    ap.add_argument("--init-from", default="",
                    help="从已有权重热启动；档位 token 的 embedding 会重随机")
    ap.add_argument("--out", default="results_gen/generator.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条，冒烟用")
    ap.add_argument("--val-gen", type=int, default=16,
                    help="每个 epoch 实采多少条验证样本测合法率；0 关闭")
    ap.add_argument("--save-last", action="store_true",
                    help="每个 epoch 另存最后一版权重（诊断 overfit 用）")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    meta = json.loads(Path(args.gen_meta).read_text())
    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print(f"编辑对 {len(rows)}（train {len(tr)} / val {len(va)}），"
          f"{meta['n_bins']} 档 x {len(meta['objectives'])} 目标")

    # 词表实扫语料得到，不手写字符表
    chars = set()
    for r in rows:
        chars.update(split_smiles(r["src"]))
        chars.update(split_smiles(r["dst"]))
    vocab = build_vocab(meta["n_bins"], chars, len(meta["objectives"]))
    print(f"词表 {len(vocab)}（其中语料字符 {len(chars)}）")

    # 前缀最长 = 1 + n_obj + 1 + len(scaffold) + 1，加上 body
    max_len = 2 * meta["max_len"] + len(meta["objectives"]) + 8
    cfg = GenConfig(vocab_size=len(vocab), n_bins=meta["n_bins"],
                    n_obj=len(meta["objectives"]), d_model=args.d_model,
                    n_layer=args.n_layer, n_head=args.n_head,
                    d_ff=4 * args.d_model, max_len=max_len)
    model = CondSmilesTransformer(cfg)
    print(f"参数量 {count_params(model)/1e6:.2f}M，max_len {max_len}，"
          f"{args.threads} threads")

    if args.init_from:
        # 架构和语料没变，只有 16 个档位 token 的**含义**变了
        # （绝对分位 -> 变化分位，见 scripts/rebin_delta.py）。
        # "照抄 SMILES 并小改"是已学到的大头，可以迁移；只有这 16 行
        # embedding 必须重随机，否则模型会按旧含义解读新条件。
        blob = torch.load(args.init_from, map_location="cpu",
                          weights_only=False)
        if blob["vocab"] != vocab:
            print(f"!! 词表不一致（旧 {len(blob['vocab'])} / 新 {len(vocab)}），"
                  "放弃热启动，从随机初始化开始")
        else:
            model.load_state_dict(blob["state"])
            bin_rows = [vocab[f"<b{j}_{v}>"]
                        for j in range(cfg.n_obj) for v in range(cfg.n_bins)]
            with torch.no_grad():
                # 必须用 __setitem__ 写回：model.tok.weight[bin_rows] 是高级
                # 索引，取出来的是**副本**，对它调 nn.init 等于什么都没做。
                model.tok.weight[bin_rows] = 0.02 * torch.randn(
                    len(bin_rows), cfg.d_model)
            print(f"热启动自 {args.init_from}（epoch {blob.get('epoch')}，"
                  f"val {blob.get('val', float('nan')):.4f}），"
                  f"重随机 {len(bin_rows)} 个档位 token 的 embedding")

    pad = vocab["<pad>"]

    def seq_len(r):
        return min(len(r["src"]) + len(r["dst"]) + 8, max_len)

    def mk(rs, bucket):
        ds = ReconDataset(rs, vocab, max_len)
        kw = dict(collate_fn=lambda b: collate(b, pad), num_workers=0)
        if bucket:
            return DataLoader(ds, batch_sampler=LengthBucketSampler(
                [seq_len(r) for r in rs], args.batch), **kw)
        return DataLoader(ds, batch_size=args.batch, shuffle=False, **kw)

    dl_tr, dl_va = mk(tr, True), mk(va, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    steps = max(args.epochs * len(dl_tr), 1)
    warm = max(int(0.03 * steps), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, hist = float("inf"), []
    for ep in range(args.epochs):
        t0 = time.time()
        trl = run_epoch(model, dl_tr, pad, opt, sched)
        val = run_epoch(model, dl_va, pad)
        q = sample_quality(model, va, vocab, args.val_gen)
        hist.append({"epoch": ep, "train": trl, "val": val,
                     "ppl": math.exp(min(val, 20)), "sec": time.time() - t0,
                     **{f"gen_{k}": v for k, v in q.items()}})
        flag = ""
        if val < best:
            best = val
            torch.save({"cfg": cfg.__dict__, "vocab": vocab,
                        "state": model.state_dict(), "gen_meta": meta,
                        "epoch": ep, "val": val}, out)
            flag = "  <- saved"
        qs = (f"  valid {q['valid']:.0%}  scaf {q['scaffold']:.0%}"
              f"  exact {q['exact']:.0%}" if q else "")
        print(f"epoch {ep:2d}  train {trl:.4f}  val {val:.4f}  "
              f"ppl {math.exp(min(val,20)):.2f}{qs}  "
              f"{time.time()-t0:.0f}s{flag}", flush=True)
        Path(str(out) + ".hist.json").write_text(json.dumps(hist, indent=2))
        if args.save_last:      # 诊断用：best-val 可能是很早的 epoch
            torch.save({"cfg": cfg.__dict__, "vocab": vocab,
                        "state": model.state_dict(), "gen_meta": meta,
                        "epoch": ep, "val": val}, str(out) + ".last.pt")

    print(f"\n最好 val loss {best:.4f}，写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
