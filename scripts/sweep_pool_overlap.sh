#!/bin/bash
# 扫 rdkit 池的重叠程度，测"routing 收益 vs action 空间有效维度"。
#
# 为什么要这个：findings §11 的第 5 条结论建立在两个点上（rdkit 2.20 /
# 生成式 1.80），而这两个点同时还差着 editor 家族、产出率、validity，
# 所以"收益跟着维度走"是论证不是辨识。这里在**同一个 editor 家族内部**
# 只动重叠程度，把它做成一条曲线。
#
# 每个变体跑 3 个训练种子：§7 测得重训噪声 0.00355，比 §11a 里所有效应都大,
# 单种子每点只会给出一片噪声散点。
#
# α=0 这一点不重跑，已有 results_base0 / results_seed1 / results_seed2。
# q007 (1.70) 和 q003 (1.97) 在生成式支的 1.80 维两侧各放一个**同族**的点：
# 如果同一个 rdkit 家族在 1.8 维附近也丢掉 bandit 的收益，
# 那"收益跟着维度走"就不再只是跨家族的论证。
#
# 协议与 rdkit 支逐字一致：200/100 种子、20 步、repeats 3、train-repeats 3。
# 单轮约 53 分钟，18 轮约 16 小时。顺序跑 —— 评估阶段本来就占满 8 核。
set -u
cd /home/ubuntu/lingollm

# 变体表按**实测维度**选点，不按 alpha 均匀取：alpha 对维度的影响极不线性
# —— 掺 1 条通用删除反应（[c:1][C;!R;H3]>>[c:1]，脱芳环甲基）就把 2.20
# 压到 1.57，因为 gsk3b/jnk3 自己的池全是加成，一条通用删除立刻让它们有了
# sa 的行为。所以均匀取 alpha 会得到一堆挤在 1.4-1.6 的冗余点，而 1.6-2.2
# 空着 —— 生成式支的 1.80 正好在那段。低维端用 alpha，
# 1.7-2.5 这段改从 qed_disjoint 的 2.50 往下降。
for v in "a100 --pool-overlap 1.0" \
         "a050 --pool-overlap 0.5" \
         "a025 --pool-overlap 0.25" \
         "q007 --pool-overlap 0.07 --qed-disjoint" \
         "q003 --pool-overlap 0.03 --qed-disjoint" \
         "qdis --qed-disjoint"; do
  set -- $v
  tag=$1; shift
  for s in 0 1 2; do
    out="results_sweep/${tag}_s${s}"
    if [ -f "$out/summary.json" ]; then
      echo "=== skip $tag seed=$s（已完成）"
      continue
    fi
    echo "=== $tag seed=$s  开始 $(date +%H:%M) ==="
    python3 -u scripts/run_experiment.py \
        --editor-kind rdkit --n-train 200 --n-test 100 \
        --repeats 3 --train-repeats 3 --max-steps 20 --workers 8 \
        --seed "$s" "$@" --out "$out" \
        > "results_sweep/${tag}_s${s}.log" 2>&1
    echo "    完成 $(date +%H:%M)  -> $out"
  done
done

echo "=== SWEEP DONE $(date +%H:%M) ==="
