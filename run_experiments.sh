#!/bin/bash
# 运行消融实验组 1, 2, 3 的批处理脚本

# 设定多个随机种子以满足统计可靠性要求（根据实际需要调整列表长度，例如扩展到 {1..10} 或 {1..20}）
SEEDS=(42 43 44) 
BASE_OUT="results/ablation"

# 创建输出根目录
mkdir -p $BASE_OUT

echo "======================================"
echo "🚀 开始执行消融实验 (Groups 1, 2, 3)"
echo "种子列表: ${SEEDS[*]}"
echo "======================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======================================"
    echo "🎯 当前执行随机种子: Seed $seed"
    echo "======================================"

    # ==========================================
    # Group 1: 基础 GA-HPO DQN
    # ==========================================
    echo ">>> [1/3] 正在运行实验组 1：基础 GA-HPO DQN (禁用 特征工程、PER、N-Step、GNN)"
    python3 train_fe_iddqn_ga_hpo.py --mode full --seed $seed \
        --disable_fe --disable_per --disable_nstep --no_gnn \
        --output_dir $BASE_OUT/ga_hpo_dqn/seed_$seed

    # ==========================================
    # Group 2: GA-HPO FE-DQN
    # ==========================================
    echo ">>> [2/3] 正在运行实验组 2：GA-HPO FE-DQN (禁用 PER、N-Step、GNN，保留特征工程)"
    python3 train_fe_iddqn_ga_hpo.py --mode full --seed $seed \
        --disable_per --disable_nstep --no_gnn \
        --output_dir $BASE_OUT/ga_hpo_fe_dqn/seed_$seed

    # ==========================================
    # Group 3: GA-HPO IDDQN
    # ==========================================
    echo ">>> [3/3] 正在运行实验组 3：GA-HPO IDDQN (禁用特征工程，保留所有高级强化学习模块)"
    python3 train_fe_iddqn_ga_hpo.py --mode full --seed $seed \
        --disable_fe \
        --output_dir $BASE_OUT/ga_hpo_iddqn/seed_$seed

done

echo ""
echo "✅ 所有指定的消融实验组 (1, 2, 3) 运行完毕！"
echo "您可以前往 $BASE_OUT 目录查看每个算法在不同种子下的运行结果。"
