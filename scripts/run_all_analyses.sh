#!/bin/bash
# 从已有的 results/raw.json + results/policies.pkl 重跑全部分析和出图。
# 不重跑实验本身（那是 run_experiment.py 的事，要几十分钟）。
#
#   bash scripts/run_all_analyses.sh [results_dir]
#
# 需要先 micromamba activate lingollm。
set -euo pipefail
cd "$(dirname "$0")/.."
RES=${1:-results}

echo "=== 1. DQN 诊断 ==="
python scripts/diagnose_dqn.py --policies "$RES/policies.pkl" \
    --out "$RES/dqn_diagnosis.json"

echo "=== 2. 预算口径对比 ==="
python scripts/compare_budgets.py --raw "$RES/raw.json" \
    --out "$RES/budget_comparison.json"

echo "=== 3. 吸收态 / 死步 ==="
python scripts/analyze_editor_depletion.py --raw "$RES/raw.json" \
    --out "$RES/editor_depletion.json"

echo "=== 4. 目标相关性与 PCA ==="
python scripts/analyze_objectives.py --out "$RES/objective_analysis.json" \
    --npz "$RES/objective_scores.npz"

echo "=== 5. action space 退化 ==="
python scripts/analyze_action_space.py --out "$RES/action_space.json"

echo "=== 6. 失败轨迹 + 分子结构图 ==="
python scripts/analyze_failures.py --raw "$RES/raw.json" \
    --policies "$RES/policies.pkl" --out "$RES/failure_cases.json"

echo "=== 7. 重试规则（吸收态的代价） ==="
python scripts/eval_masking.py --policies "$RES/policies.pkl" \
    --out "$RES/masking.json" --raw-out "$RES/masking_raw.json"

echo "=== 8. 训练种子方差 + escape-bit A/B ==="
# 需要先跑出多个种子的 results 目录；没有就跳过，不让整个脚本挂掉
if [ -f results_seed1/raw.json ] && [ -f results_seed2/raw.json ]; then
    python scripts/compare_seeds.py \
        --runs results_base0 results_seed1 results_seed2 \
        --escape results_escape0:results_base0 results_escape1:results_seed1 \
        --out "$RES/seed_variance.json"
else
    echo "跳过（没有多种子 run，先跑 run_experiment.py --seed 1/2）"
fi

echo "=== 9. 出图 ==="
python scripts/plot_results.py --raw "$RES/raw.json" --out figures
python scripts/plot_diagnostics.py --results "$RES" --out figures

echo
echo "全部完成。结论汇总见 docs/findings.md，术语表见 docs/glossary.md"
