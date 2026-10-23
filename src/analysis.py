"""配对比较的共用工具。run_experiment.py 的汇总和各分析脚本都从这里取。

核心是两种**预算口径**：

  1. editor 调用口径：横轴 = step 数。这是 proposal 写的"matched proposal budget"，
     但它有一个混淆——不同 editor 的合格产出率差一倍（gsk3b/jnk3 接近 100%，
     qed/sa 约 50%），所以偏好高产 editor 的 policy 在同样的 step 数下
     自动拿到更多候选分子。

  2. 合格候选口径：横轴 = 累计合格候选数。这才是真正对齐"搜索了多少分子"的
     预算，也是 oracle 调用的实际来源。

两种口径下结论不一致时，必须两个都报——这个不一致本身就是结果。
"""
from __future__ import annotations

import numpy as np


def episode_key(r: dict):
    """配对的单位：同一个种子分子 + 同一个 episode 随机种子。"""
    return (r["smiles"], r["episode_seed"])


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------
# 取值：两种口径
# --------------------------------------------------------------------------
def hv_at_step(r: dict, x: int) -> float | None:
    """第 x 次 editor 调用之后的 HV 增益。x 超过轨迹长度则取终点。"""
    c = r["curve"]
    if x < 0:
        return None
    return float(c[min(x, len(c) - 1)])


def hv_at_eligible(r: dict, x: float) -> float | None:
    """累计产出 x 个合格候选时的 HV 增益，在 step 之间线性插值。

    返回 None 表示这条轨迹**从未**攒够 x 个候选——这时不能外推，
    否则等于把"产出率低"伪装成"HV 低"，正好抹掉我们要测的效应。
    """
    if "eligible_curve" not in r:
        return None
    xs = np.asarray(r["eligible_curve"], dtype=float)
    ys = np.asarray(r["curve"], dtype=float)
    L = min(len(xs), len(ys))
    xs, ys = xs[:L], ys[:L]
    if xs[-1] < x:
        return None
    return float(np.interp(x, xs, ys))


# --------------------------------------------------------------------------
# 配对差
# --------------------------------------------------------------------------
def common_support(rows: list[dict], x: float) -> set:
    """所有 policy 都攒够 x 个合格候选的那些 episode。

    为什么必须这样做：按 policy 各自丢弃够不着的 episode 会引入选择偏差——
    被丢掉的恰恰是产出率低的轨迹，而产出率正是我们想控制的那个变量。
    结果就是"控制产出率"的比较反而被产出率污染。
    固定一个公共子集，所有 pair 用同一批 episode，比较才干净。
    """
    by_key: dict = {}
    policies = {r["policy"] for r in rows}
    for r in rows:
        k = episode_key(r)
        by_key.setdefault(k, {})[r["policy"]] = r["eligible_curve"][-1]
    return {k for k, d in by_key.items()
            if len(d) == len(policies) and min(d.values()) >= x}


def coverage_at(rows: list[dict], x: float) -> dict:
    """每个 policy 有多少比例的 episode 能攒够 x 个候选，外加公共支撑集大小。"""
    out = {}
    for n in {r["policy"] for r in rows}:
        sub = [r for r in rows if r["policy"] == n]
        out[n] = float(np.mean([r["eligible_curve"][-1] >= x for r in sub]))
    n_ep = len({episode_key(r) for r in rows})
    out["_common"] = len(common_support(rows, x)) / max(n_ep, 1)
    return out


def pick_candidate_budget(rows: list[dict], min_common: float = 0.8,
                          step: int = 5) -> int:
    """取公共支撑覆盖率仍 >= min_common 的最大候选预算。

    预算越大区分度越高，但够得着的 episode 越少；这个函数就是在两者之间卡一个点。
    """
    for thresh in (min_common, 0.6, 0.4):
        best = 0
        for x in range(step, 200, step):
            if coverage_at(rows, x)["_common"] >= thresh:
                best = x
            else:
                break
        if best:
            return best
    # 连最小预算点都覆盖不足（极小的 --quick 跑），退回最小值并由调用方报覆盖率
    return step


def paired_diff(rows: list[dict], a: str, b: str, value, restrict: set | None = None):
    """policy a 减 policy b，按 (分子, episode seed) 配对。

    value: 一个 r -> float|None 的函数。任一侧为 None 的 episode 整对丢弃，
    并把丢弃数量返回，便于在表里注明"这个预算点上有多少 episode 够不着"。
    restrict: 只用这批 episode（公共支撑集），避免各 pair 用不同子集。
    """
    va = {episode_key(r): value(r) for r in rows if r["policy"] == a}
    vb = {episode_key(r): value(r) for r in rows if r["policy"] == b}
    keys = sorted(set(va) & set(vb))
    if restrict is not None:
        keys = [k for k in keys if k in restrict]
    kept = [k for k in keys if va[k] is not None and vb[k] is not None]
    d = np.array([va[k] - vb[k] for k in kept], dtype=float)
    return d, len(keys) - len(kept)


def summarize_pair(rows, a, b, value, label="", restrict=None):
    d, dropped = paired_diff(rows, a, b, value, restrict=restrict)
    if len(d) == 0:
        return {"a": a, "b": b, "label": label, "n": 0, "dropped": dropped}
    lo, hi = bootstrap_ci(d)
    return {
        "a": a, "b": b, "label": label,
        "n": int(len(d)), "dropped": int(dropped),
        "mean": float(d.mean()), "lo": lo, "hi": hi,
        "wins": int((d > 0).sum()),
        "sig": bool(lo > 0 or hi < 0),
    }


def print_pair_table(res: list[dict], title: str) -> None:
    print(f"\n{title}")
    for r in res:
        if r["n"] == 0:
            print(f"  {r['a']:7}-{r['b']:11} (无可配对 episode)")
            continue
        sig = "*" if r["sig"] else " "
        drop = f"  丢弃{r['dropped']}" if r["dropped"] else ""
        print(f"  {r['a']:7}-{r['b']:11}{r['mean']:+.5f}  "
              f"[{r['lo']:+.5f}, {r['hi']:+.5f}]{sig}  "
              f"赢 {r['wins']}/{r['n']}{drop}  {r['label']}")


def participation_ratio(M: np.ndarray) -> float:
    """奇异值的参与比 = 有效维度。M 是 [editor, objective] 的平均增益矩阵。"""
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv.sum() ** 2 / max((sv ** 2).sum(), 1e-12))


def objective_scales(pairs_path: str) -> np.ndarray | None:
    """各目标**可达变化**的标准差，用来给 participation_ratio 的输入归一。

    不归一的话，dy 尺度最大的目标会独占奇异值：实测同骨架编辑对的 dy 本身
    原始口径只有 1.51/4，按维归一后是 3.45/4（qed 的 dy 标准差 0.137，
    是 jnk3 的 4.2 倍）。于是即便四个 action 彼此正交，也会被测成 ~1.5 维。

    归一化因子取自编辑对数据，是环境的性质、与被测 editor 无关，所以
    RDKit 池和各版生成器可以直接对读。3.45 也是这个统计量的真实上限 ——
    比 4.0 更该拿来当参照。
    """
    import json
    from pathlib import Path
    p = Path(pairs_path)
    if not p.exists():
        return None
    dy = np.array([json.loads(l)["dy"] for l in p.read_text().splitlines()])
    s = dy.std(axis=0)
    return s if np.all(s > 0) else None
