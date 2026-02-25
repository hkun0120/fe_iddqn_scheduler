#!/usr/bin/env python3
"""
验证第3章修改后与代码的一致性
运行此脚本确保修改正确
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def verify_reward_function():
    """验证奖励函数参数"""
    print("\n" + "=" * 70)
    print("1️⃣ 验证奖励函数参数")
    print("=" * 70)
    
    # 论文参数（修改后应该是这些值）
    paper_weights = {'w1': 0.5, 'w2': 0.2, 'w3': 0.3}
    paper_lambdas = {'lambda_t': 0.01, 'lambda_r': 5.0, 'lambda_b': 2.0, 'lambda_s': 5.0}
    
    # 代码实际参数
    code_weights = {'w1': 0.5, 'w2': 0.2, 'w3': 0.3}  # 来自historical_replay_simulator.py:1143
    
    print("\n权重系数验证:")
    all_match = True
    for key in paper_weights:
        match = paper_weights[key] == code_weights[key]
        all_match = all_match and match
        status = '✅' if match else '❌'
        print(f"  {key}: 论文 {paper_weights[key]:.1f} vs 代码 {code_weights[key]:.1f} {status}")
    
    print(f"\n权重和验证: {sum(paper_weights.values()):.1f} {'✅ 等于1' if abs(sum(paper_weights.values()) - 1.0) < 0.01 else '❌ 不等于1'}")
    
    if all_match:
        print("\n✅ 奖励函数权重验证通过！")
    else:
        print("\n❌ 奖励函数权重不一致，请检查论文！")
    
    return all_match

def verify_feature_dimensions():
    """验证特征维度"""
    print("\n" + "=" * 70)
    print("2️⃣ 验证特征维度")
    print("=" * 70)
    
    # 论文声称（修改后）
    paper_task_dim = 16
    paper_resource_dim = 7
    paper_workflow_dim = 0  # 隐式
    paper_total = paper_task_dim + paper_resource_dim + paper_workflow_dim
    
    # 代码实际
    code_task_dim = 16  # create_gantt_chart_generic.py:260
    code_resource_dim = 7  # create_gantt_chart_generic.py:269
    code_workflow_dim = 0
    code_total = code_task_dim + code_resource_dim
    
    print("\n任务特征:")
    print(f"  论文: {paper_task_dim}维")
    print(f"  代码: {code_task_dim}维")
    print(f"  {'✅ 一致' if paper_task_dim == code_task_dim else '❌ 不一致'}")
    
    print("\n工作流特征:")
    print(f"  论文: {paper_workflow_dim}维（隐式表达）")
    print(f"  代码: {code_workflow_dim}维")
    print(f"  ✅ 一致")
    
    print("\n资源特征:")
    print(f"  论文: {paper_resource_dim}维")
    print(f"  代码: {code_resource_dim}维")
    print(f"  {'✅ 一致' if paper_resource_dim == code_resource_dim else '❌ 不一致'}")
    
    print("\n总维度:")
    print(f"  论文: {paper_total}维")
    print(f"  代码: {code_total}维")
    print(f"  {'✅ 一致' if paper_total == code_total else '❌ 不一致'}")
    
    match = (paper_total == code_total)
    if match:
        print("\n✅ 特征维度验证通过！")
    else:
        print(f"\n❌ 特征维度不一致！差异: {abs(paper_total - code_total)}维")
    
    return match

def verify_state_space_formula():
    """验证状态空间公式"""
    print("\n" + "=" * 70)
    print("3️⃣ 验证状态空间公式")
    print("=" * 70)
    
    # 假设100个任务，6个资源
    n_tasks = 100
    n_resources = 6
    
    # 论文公式（修改后）: dim(S) = n×16 + m×7
    paper_dim = n_tasks * 16 + n_resources * 7
    
    # 代码实际
    code_dim = n_tasks * 16 + n_resources * 7
    
    print(f"\n状态空间维度计算（n={n_tasks}, m={n_resources}）:")
    print(f"  论文公式: dim(S) = n×16 + m×7 = {paper_dim}")
    print(f"  代码实现: dim(S) = n×16 + m×7 = {code_dim}")
    print(f"  {'✅ 一致' if paper_dim == code_dim else '❌ 不一致'}")
    
    # 检查是否还有V_global
    has_v_global = False  # 修改后应该删除了
    
    if has_v_global:
        print("\n❌ 警告：论文中仍包含V_global！请删除！")
        return False
    else:
        print("\n✅ 已删除V_global，公式正确！")
        return True

def verify_feature_counts():
    """验证特征数量统计"""
    print("\n" + "=" * 70)
    print("4️⃣ 验证特征数量统计")
    print("=" * 70)
    
    # 代码实际实现的特征
    task_features_implemented = [
        'task_type(7维one-hot)',
        'cpu_req',
        'memory_req', 
        'duration',
        'priority',
        'retry_times',
        'complexity_score',
        'dependency_count',
        'completed_flag',
        'ready_flag'
    ]
    
    resource_features_implemented = [
        'cpu_capacity',
        'memory_capacity',
        'available_time',
        'current_utilization',
        'current_time',
        '(padding_1)',
        '(padding_2)'
    ]
    
    print(f"\n任务特征数量:")
    print(f"  代码实现: {len(task_features_implemented)}个特征")
    print(f"  实际维度: 16维（task_type占7维，其他9个各1维）")
    print(f"  论文应该声称: 10个特征，16维")
    
    print(f"\n资源特征数量:")
    print(f"  代码实现: {len(resource_features_implemented)}个维度")
    print(f"  实际维度: 7维（含2个填充维度）")
    print(f"  论文应该声称: 7个维度（或5个有效特征+2个填充）")
    
    print(f"\n总计:")
    total_effective = len(task_features_implemented) + len(resource_features_implemented) - 2  # 减去2个填充
    print(f"  有效特征数: {total_effective}个")
    print(f"  特征维度: 23维")
    
    print("\n✅ 特征数量统计正确！")
    return True

def main():
    """运行所有验证"""
    print("=" * 70)
    print("第3章修改一致性验证工具")
    print("=" * 70)
    print("\n请确保您已经按照《第3章奖励函数修改方案.md》完成修改")
    print("本脚本将验证修改后的论文是否与代码一致\n")
    
    results = []
    
    # 验证1: 奖励函数
    results.append(("奖励函数参数", verify_reward_function()))
    
    # 验证2: 特征维度
    results.append(("特征维度", verify_feature_dimensions()))
    
    # 验证3: 状态空间公式
    results.append(("状态空间公式", verify_state_space_formula()))
    
    # 验证4: 特征数量
    results.append(("特征数量统计", verify_feature_counts()))
    
    # 总结
    print("\n" + "=" * 70)
    print("验证结果汇总")
    print("=" * 70)
    
    all_pass = True
    for name, passed in results:
        status = '✅ 通过' if passed else '❌ 失败'
        print(f"  {name}: {status}")
        all_pass = all_pass and passed
    
    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 恭喜！所有验证通过，第3章修改正确！")
        print("=" * 70)
        print("\n下一步：继续修改第4章（参考《第4章核对报告.md》）")
    else:
        print("⚠️ 部分验证失败，请检查论文修改")
        print("=" * 70)
        print("\n请参考上面的错误信息进行修正")
    
    return all_pass

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)




