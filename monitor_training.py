#!/usr/bin/env python3
"""
监控FE-IDDQN训练进度
"""

import time
import os
import re
from datetime import datetime

def monitor_training():
    """监控训练进度"""
    log_file = 'fe_iddqn_training_system/training_improved_parallel.log'
    
    print("=" * 80)
    print("FE-IDDQN 训练进度监控")
    print("=" * 80)
    
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    # 读取日志文件
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 分析训练进度
    episodes = []
    rewards = []
    makespans = []
    resource_utils = []
    parallel_effs = []
    
    for line in lines:
        # 提取Episode信息
        if "Episode" in line and "Reward=" in line:
            # 解析Episode信息
            episode_match = re.search(r'Episode (\d+)', line)
            reward_match = re.search(r'Reward=([\d.-]+)', line)
            makespan_match = re.search(r'Makespan=([\d.-]+)', line)
            resource_util_match = re.search(r'Resource_Util=([\d.-]+)', line)
            parallel_eff_match = re.search(r'Parallel_Eff=([\d.-]+)', line)
            
            if episode_match:
                episode_num = int(episode_match.group(1))
                episodes.append(episode_num)
                
                if reward_match:
                    rewards.append(float(reward_match.group(1)))
                if makespan_match:
                    makespans.append(float(makespan_match.group(1)))
                if resource_util_match:
                    resource_utils.append(float(resource_util_match.group(1)))
                if parallel_eff_match:
                    parallel_effs.append(float(parallel_eff_match.group(1)))
    
    # 显示训练进度
    if episodes:
        latest_episode = max(episodes)
        print(f"✅ 最新Episode: {latest_episode}")
        
        if rewards:
            latest_reward = rewards[-1]
            avg_reward = sum(rewards[-10:]) / len(rewards[-10:]) if len(rewards) >= 10 else sum(rewards) / len(rewards)
            print(f"✅ 最新奖励: {latest_reward:.2f}")
            print(f"✅ 平均奖励(最近10个): {avg_reward:.2f}")
        
        if makespans:
            latest_makespan = makespans[-1]
            avg_makespan = sum(makespans[-10:]) / len(makespans[-10:]) if len(makespans) >= 10 else sum(makespans) / len(makespans)
            print(f"✅ 最新Makespan: {latest_makespan:.2f}")
            print(f"✅ 平均Makespan(最近10个): {avg_makespan:.2f}")
        
        if resource_utils:
            latest_resource_util = resource_utils[-1]
            avg_resource_util = sum(resource_utils[-10:]) / len(resource_utils[-10:]) if len(resource_utils) >= 10 else sum(resource_utils) / len(resource_utils)
            print(f"✅ 最新资源利用率: {latest_resource_util:.4f}")
            print(f"✅ 平均资源利用率(最近10个): {avg_resource_util:.4f}")
        
        if parallel_effs:
            latest_parallel_eff = parallel_effs[-1]
            avg_parallel_eff = sum(parallel_effs[-10:]) / len(parallel_effs[-10:]) if len(parallel_effs) >= 10 else sum(parallel_effs) / len(parallel_effs)
            print(f"✅ 最新并行效率: {latest_parallel_eff:.3f}")
            print(f"✅ 平均并行效率(最近10个): {avg_parallel_eff:.3f}")
        
        # 分析改进趋势
        print("\n" + "=" * 50)
        print("改进趋势分析")
        print("=" * 50)
        
        if len(episodes) >= 20:
            # 对比前10个和后10个Episode
            early_rewards = rewards[:10]
            recent_rewards = rewards[-10:]
            
            early_avg_reward = sum(early_rewards) / len(early_rewards)
            recent_avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            reward_improvement = (recent_avg_reward - early_avg_reward) / early_avg_reward * 100 if early_avg_reward != 0 else 0
            
            print(f"奖励改进: {reward_improvement:.2f}%")
            
            if parallel_effs and len(parallel_effs) >= 20:
                early_parallel_eff = sum(parallel_effs[:10]) / len(parallel_effs[:10])
                recent_parallel_eff = sum(parallel_effs[-10:]) / len(parallel_effs[-10:])
                
                parallel_improvement = (recent_parallel_eff - early_parallel_eff) / early_parallel_eff * 100 if early_parallel_eff != 0 else 0
                
                print(f"并行效率改进: {parallel_improvement:.2f}%")
        
        # 判断训练状态
        print("\n" + "=" * 50)
        print("训练状态评估")
        print("=" * 50)
        
        if parallel_effs and len(parallel_effs) >= 10:
            recent_parallel_eff = sum(parallel_effs[-10:]) / len(parallel_effs[-10:])
            
            if recent_parallel_eff > 0.4:
                print("🎉 优秀！并行效率超过40%")
            elif recent_parallel_eff > 0.3:
                print("✅ 良好！并行效率超过30%")
            elif recent_parallel_eff > 0.2:
                print("⚠️  一般，并行效率需要提升")
            else:
                print("❌ 较差，并行效率过低")
        
        if resource_utils and len(resource_utils) >= 10:
            recent_resource_util = sum(resource_utils[-10:]) / len(resource_utils[-10:])
            
            if recent_resource_util > 0.3:
                print("🎉 优秀！资源利用率超过30%")
            elif recent_resource_util > 0.2:
                print("✅ 良好！资源利用率超过20%")
            else:
                print("⚠️  资源利用率需要提升")
    
    else:
        print("⏳ 训练尚未开始或数据加载中...")
    
    # 显示最新日志
    print("\n" + "=" * 50)
    print("最新日志")
    print("=" * 50)
    
    # 显示最后10行日志
    recent_lines = lines[-10:] if len(lines) >= 10 else lines
    for line in recent_lines:
        print(line.strip())
    
    print("\n" + "=" * 80)
    print("监控完成")
    print("=" * 80)

if __name__ == "__main__":
    monitor_training()
