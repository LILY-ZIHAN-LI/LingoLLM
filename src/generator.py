"""条件 SMILES transformer：(scaffold, 属性档位) -> 完整分子 SMILES。

对应 proposal §3.1 的 compact generator：token cross-entropy 的重建目标，
推理时把某一个目标的档位往上推一格、其余保持不变 —— 这就得到四个
objective-conditioned 的生成式 editor，而**权重是共享的**
（proposal: "Experts may share one frozen generative backbone with different
objective prompts or conditioning; they need not be separately trained models."）。

和 RDKit 编辑池的关键区别：
  RDKit 池的 action 空间 = 反应模板集合，两个 editor 可能字面共用同一个池
  （见 docs/findings.md §1b，qed/sa 在 70/100 个种子上提议完全相同）。
  这里 action 空间 = 条件向量，四个 editor 必然对应四个不同的条件，
  是否真的产生不同的分布是可检验的 —— 这正是 proposal 要求的
  generator-quality diagnostic。

架构（CPU 可训）：decoder-only，条件作为前缀 token。
  [BOS] <bin_gsk3b> <bin_jnk3> <bin_qed> <bin_sa> <SCAF> scaffold... <MOL> smiles... [EOS]
loss 只加在 <MOL> 之后的位置上 —— 前缀是条件，不是要学的分布。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 两字符/三字符的元素与记号必须先于单字符匹配，否则 'Cl' 会被切成 'C','l'。
# %10-%99 是双位环闭合编号，不先吃掉就会被当成两个环编号，重建出错误的分子。
MULTI = ("Cl", "Br", "Si", "Se", "se", "@@", "+2", "-2", "+3", "-3") + tuple(
    f"%{i}" for i in range(10, 100))
SPECIAL = ["<pad>", "<bos>", "<eos>", "<src>", "<mol>", "<unk>"]


def split_smiles(smi: str) -> list[str]:
    """字符级切分，但多字符记号整体保留。不依赖词表，供扫语料建词表用。"""
    out, i = [], 0
    while i < len(smi):
        for m in MULTI:
            if smi.startswith(m, i):
                out.append(m)
                i += len(m)
                break
        else:
            out.append(smi[i])
            i += 1
    return out


def build_vocab(n_bins: int, chars, n_obj: int = 4) -> dict[str, int]:
    """chars 来自语料实扫（见 scripts/train_generator.py），不靠手写字符表猜。"""
    toks = list(SPECIAL)
    toks += [f"<b{j}_{v}>" for j in range(n_obj) for v in range(n_bins)]
    toks += sorted(set(chars))
    out: dict[str, int] = {}
    for t in toks:
        if t not in out:
            out[t] = len(out)
    return out


def tokenize(smi: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(t, unk) for t in split_smiles(smi)]


def detokenize(ids: list[int], inv: dict[int, str]) -> str:
    out = []
    for i in ids:
        t = inv.get(i, "")
        if t in ("<eos>", "<pad>"):
            break
        if t.startswith("<"):
            continue
        out.append(t)
    return "".join(out)


@dataclass
class GenConfig:
    vocab_size: int
    n_bins: int = 4
    n_obj: int = 4
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    d_ff: int = 1024
    max_len: int = 260
    dropout: float = 0.1


class CondSmilesTransformer(nn.Module):
    """decoder-only，causal mask，条件走前缀 token。

    参数量 ~ n_layer * 12 * d_model^2。默认 6x256 约 4.8M —— proposal 写的是
    10-30M，但那是按一块 16-32GB GPU 规划的；本机是 CPU-only c7i.2xlarge，
    所以取下限以下。这是需要在报告里明写的 deviation（见 findings §9）。
    """

    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_head, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.body = nn.TransformerEncoder(layer, cfg.n_layer)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        p = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(p)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.body(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))

    @torch.no_grad()
    def sample_batch(self, prefix: list[int], n: int, max_new: int, eos: int,
                     temperature: float = 1.0, top_k: int = 0,
                     generator: torch.Generator | None = None
                     ) -> list[list[int]]:
        """同一前缀采 n 条，合成一个 batch。

        CPU 上这是 editor 的主要开销来源：逐条采样时每步是 batch=1 的
        前向，矩阵太小，时间花在访存和算子调度上而不是 FLOPs。
        合成 batch=n 后每步只调度一次。

        确定性：整批共用一个 generator（由 GenEditor 按
        (parent, editor名, seed) 导出种子）。interfaces.Editor 要求的是
        "相同 (smiles, seed) 给相同输出"，这一点仍然成立；只是具体采到
        哪 n 个分子与逐条采样时不同。
        """
        self.eval()
        x = torch.tensor([prefix], dtype=torch.long).repeat(n, 1)
        out: list[list[int]] = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool)
        for _ in range(max_new):
            if x.shape[1] >= self.cfg.max_len:
                break
            logits = self(x)[:, -1] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, -1), 1,
                                    generator=generator).squeeze(1)
            for i in range(n):
                if not done[i]:
                    t = int(nxt[i])
                    if t == eos:
                        done[i] = True
                    else:
                        out[i].append(t)
            if bool(done.all()):
                break
            # 已结束的行继续喂 eos，不影响它的输出（上面被 done 挡住）
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        return out

    @torch.no_grad()
    def sample(self, prefix: list[int], max_new: int, eos: int,
               temperature: float = 1.0, top_k: int = 0,
               generator: torch.Generator | None = None) -> list[int]:
        self.eval()
        x = torch.tensor([prefix], dtype=torch.long)
        out: list[int] = []
        for _ in range(max_new):
            if x.shape[1] >= self.cfg.max_len:
                break
            logits = self(x)[0, -1] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits = logits.masked_fill(logits < v[-1], float("-inf"))
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1,
                                        generator=generator))
            if nxt == eos:
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
        return out


def make_prefix(bins: list[int], src_ids: list[int],
                vocab: dict[str, int]) -> list[int]:
    """[bos] <b0_v> .. <b3_v> <src> ...当前分子... <mol>

    条件是**当前分子**本身，不是它的 scaffold。两个原因：
      1. proposal §3.1 规定 expert 看到的是 "current molecule/scaffold,
         current property scores, and an objective-specific target"；
      2. 只给 scaffold 的话，模型会丢掉当前分子的全部取代基，每次调用等于跳到
         一个任意同骨架类似物 —— 那样轨迹不再累积，routing 也就无从谈起。
         冒烟实验里这一版的 scaffold 保持率是 0%。
    以当前分子为条件时，模型只要学会"大部分照抄、按目标档位改一点"，
    保 scaffold 变成容易学的行为而不是碰运气。
    """
    ids = [vocab["<bos>"]]
    ids += [vocab[f"<b{j}_{v}>"] for j, v in enumerate(bins)]
    ids += [vocab["<src>"]] + src_ids + [vocab["<mol>"]]
    return ids


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
