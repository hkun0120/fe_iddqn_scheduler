#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运维调度系统延时任务分析报告生成脚本
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义清华大学学术风格配色
PURPLE_PRIMARY = '#6B3FA0'  # 主紫色
PURPLE_DARK = '#4A2C6A'     # 深紫色
PURPLE_LIGHT = '#9B6DD4'    # 浅紫色
WHITE = '#FFFFFF'
GRAY_LIGHT = '#F5F5F5'
GRAY_DARK = '#333333'

# 输出目录
OUTPUT_DIR = '/home/z/my-project/download/'

def create_time_trend_chart():
    """创建时间趋势图"""
    # 时间趋势数据
    time_data = [
        ('2026-01-06 00:00', 197, 107.66),
        ('2026-01-06 01:00', 2, 41.00),
        ('2026-01-06 02:00', 6, 1699.67),
        ('2026-01-06 03:00', 18, 34.83),
        ('2026-01-06 04:00', 13, 770.62),
        ('2026-01-06 05:00', 4, 12050.25),
        ('2026-01-06 06:00', 116, 809.14),
        ('2026-01-06 07:00', 10, 41.00),
        ('2026-01-06 08:00', 5, 335.20),
        ('2026-01-06 09:00', 2, 41.00),
        ('2026-01-06 10:00', 15, 13.00),
        ('2026-01-06 11:00', 14, 5786.93),
        ('2026-01-06 12:00', 2, 41.00),
        ('2026-01-06 13:00', 9, 15468.56),
        ('2026-01-06 14:00', 4, 41.00),
        ('2026-01-06 15:00', 2, 41.00),
        ('2026-01-06 16:00', 1, 41.00),
        ('2026-01-06 17:00', 1, 19249.00),
        ('2026-01-06 18:00', 2, 4176.50),
        ('2026-01-06 19:00', 2, 41.00),
        ('2026-01-06 20:00', 16, 12.06),
        ('2026-01-06 21:00', 1, 41.00),
        ('2026-01-06 22:00', 1, 41.00),
        ('2026-01-06 23:00', 12, 63719.67),
        ('2026-01-07 00:00', 252, 85.51),
        ('2026-01-07 01:00', 336, 102.95),
        ('2026-01-07 02:00', 228, 27.09),
        ('2026-01-07 03:00', 1352, 134.04),
    ]
    
    # 解析数据
    times = [datetime.strptime(d[0], '%Y-%m-%d %H:%M') for d in time_data]
    task_counts = [d[1] for d in time_data]
    avg_waits = [d[2] for d in time_data]
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 设置背景色
    fig.patch.set_facecolor(WHITE)
    ax1.set_facecolor(WHITE)
    
    # 绘制任务数量柱状图
    bars = ax1.bar(times, task_counts, width=0.04, color=PURPLE_PRIMARY, alpha=0.7, label='延时任务数')
    
    # 高亮峰值
    max_idx = task_counts.index(max(task_counts))
    bars[max_idx].set_color(PURPLE_DARK)
    bars[max_idx].set_edgecolor('#FF6B6B')
    bars[max_idx].set_linewidth(2)
    
    # 标注峰值
    ax1.annotate(f'峰值: {task_counts[max_idx]}个任务\n{times[max_idx].strftime("%m/%d %H:00")}',
                xy=(times[max_idx], task_counts[max_idx]),
                xytext=(times[max_idx] + timedelta(hours=2), task_counts[max_idx] + 100),
                fontsize=11, fontweight='bold', color=PURPLE_DARK,
                arrowprops=dict(arrowstyle='->', color=PURPLE_DARK, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=WHITE, edgecolor=PURPLE_PRIMARY))
    
    ax1.set_xlabel('时间', fontsize=12, fontweight='bold')
    ax1.set_ylabel('延时任务数', fontsize=12, fontweight='bold', color=PURPLE_DARK)
    ax1.tick_params(axis='y', labelcolor=PURPLE_DARK)
    
    # 创建第二个Y轴（平均等待时长）
    ax2 = ax1.twinx()
    line = ax2.plot(times, avg_waits, color='#FF6B6B', linewidth=2, marker='o', 
                    markersize=4, label='平均等待时长(秒)', alpha=0.8)
    ax2.set_ylabel('平均等待时长(秒)', fontsize=12, fontweight='bold', color='#FF6B6B')
    ax2.tick_params(axis='y', labelcolor='#FF6B6B')
    
    # 设置X轴格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:00'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.xticks(rotation=45, ha='right')
    
    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    
    # 标题
    plt.title('48小时延时任务趋势分析 (2026/04/06 - 2026/04/07)', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'time_trend.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f'时间趋势图已保存: {output_path}')
    return output_path

def create_host_comparison_chart():
    """创建主机堆积对比图"""
    # 4月7日主机数据（Top 10）
    host_data = [
        ('10.39.185.5:1234', 2048, 113.75),
        ('10.39.186.37:1234', 78, 98.81),
        ('10.39.184.62:1234', 75, 98.65),
        ('10.39.186.54:1234', 72, 99.71),
        ('10.39.187.57:1234', 62, 12418.21),
        ('10.39.187.56:1234', 56, 94.20),
        ('10.39.187.178:1234', 38, 92.61),
        ('10.84.18.73:6678', 36, 6243.36),
        ('10.39.187.172:1234', 30, 94.50),
        ('10.39.185.12:1234', 20, 84.60),
    ]
    
    hosts = [d[0].split(':')[0] for d in host_data]  # 只显示IP
    tasks = [d[1] for d in host_data]
    avg_waits = [d[2] for d in host_data]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    
    # 创建颜色渐变
    colors = [PURPLE_DARK] + [PURPLE_PRIMARY] + [PURPLE_LIGHT] * 8
    
    # 横向条形图
    y_pos = np.arange(len(hosts))
    bars = ax.barh(y_pos, tasks, color=colors, edgecolor='white', linewidth=0.5)
    
    # 高亮异常主机
    bars[0].set_color('#FF6B6B')  # 最高堆积主机
    bars[0].set_edgecolor(PURPLE_DARK)
    bars[0].set_linewidth(2)
    
    # 添加数值标签
    for i, (bar, task, wait) in enumerate(zip(bars, tasks, avg_waits)):
        width = bar.get_width()
        if i == 0:
            # 异常主机特殊标注
            ax.text(width + 30, bar.get_y() + bar.get_height()/2,
                   f'{task} (均值26倍)',
                   ha='left', va='center', fontsize=10, fontweight='bold', color='#FF6B6B')
        else:
            ax.text(width + 10, bar.get_y() + bar.get_height()/2,
                   f'{task}',
                   ha='left', va='center', fontsize=9, color=GRAY_DARK)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hosts, fontsize=10)
    ax.invert_yaxis()  # 最大的在上面
    ax.set_xlabel('延时任务数', fontsize=12, fontweight='bold')
    ax.set_title('主机延时任务堆积对比 (Top 10)', fontsize=14, fontweight='bold', pad=15)
    
    # 添加均值参考线
    mean_val = np.mean(tasks[1:])  # 排除异常值计算均值
    ax.axvline(x=mean_val, color=PURPLE_LIGHT, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(mean_val + 5, len(hosts) - 0.5, f'均值: {mean_val:.0f}', 
            fontsize=9, color=PURPLE_LIGHT, fontweight='bold')
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'host_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f'主机堆积对比图已保存: {output_path}')
    return output_path

def create_wait_distribution_chart():
    """创建等待时长分布饼图"""
    # 等待时长分布数据
    labels = ['1m-5m', '5s-30s', '31s-1m', '>5m']
    sizes = [53.87, 25.31, 13.23, 7.59]
    counts = [1413, 664, 347, 199]
    
    # 颜色配置（紫色系渐变）
    colors = [PURPLE_DARK, PURPLE_PRIMARY, PURPLE_LIGHT, '#C9A8E8']
    explode = (0.05, 0, 0, 0.08)  # 突出主要区间
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       wedgeprops=dict(edgecolor=WHITE, linewidth=2),
                                       textprops=dict(fontsize=11))
    
    # 设置自动百分比文本样式
    for autotext in autotexts:
        autotext.set_color(WHITE)
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # 添加任务数量标注
    legend_labels = [f'{label}: {count}个任务 ({size}%)' 
                     for label, count, size in zip(labels, counts, sizes)]
    ax.legend(wedges, legend_labels, title='等待时长分布', loc='center left',
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    ax.set_title('任务等待时长分布特征', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'wait_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f'等待时长分布图已保存: {output_path}')
    return output_path

def create_resource_analysis_chart():
    """创建资源维度分析图"""
    # 6678端口主机数据（疑似调度节点异常）
    special_hosts = [
        ('10.84.18.73:6678', 36, 5611.42),
        ('10.84.18.74:6678', 9, 18636.78),
    ]
    
    # 普通主机对比数据
    normal_hosts = [
        ('10.39.185.5:1234', 2048, 113.75),
        ('10.39.186.37:1234', 78, 98.81),
        ('10.39.184.62:1234', 75, 98.65),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(WHITE)
    
    # 左图：平均等待时长对比
    ax1 = axes[0]
    ax1.set_facecolor(WHITE)
    
    all_hosts = special_hosts + normal_hosts[:3]
    names = [h[0].split(':')[0] for h in all_hosts]
    waits = [h[2] for h in all_hosts]
    
    colors = ['#FF6B6B', '#FF6B6B', PURPLE_DARK, PURPLE_PRIMARY, PURPLE_LIGHT]
    bars = ax1.bar(range(len(names)), waits, color=colors, edgecolor='white', linewidth=1)
    
    # 标注异常值
    for i, (bar, wait) in enumerate(zip(bars, waits)):
        if wait > 1000:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{wait:.0f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#FF6B6B')
    
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('平均等待时长(秒)', fontsize=11, fontweight='bold')
    ax1.set_title('平均等待时长对比', fontsize=12, fontweight='bold')
    ax1.axhline(y=100, color=PURPLE_LIGHT, linestyle='--', alpha=0.7, label='正常阈值')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 右图：端口类型分布
    ax2 = axes[1]
    ax2.set_facecolor(WHITE)
    
    port_data = {
        '1234端口': 2522,
        '6678端口': 45,
        '2234端口': 9,
        '3234端口': 5
    }
    
    colors_pie = [PURPLE_PRIMARY, '#FF6B6B', PURPLE_LIGHT, '#C9A8E8']
    wedges, texts, autotexts = ax2.pie(port_data.values(), labels=port_data.keys(),
                                        colors=colors_pie, autopct='%1.1f%%',
                                        startangle=90, explode=(0, 0.1, 0, 0),
                                        wedgeprops=dict(edgecolor=WHITE, linewidth=2))
    
    for autotext in autotexts:
        autotext.set_color(WHITE)
        autotext.set_fontweight('bold')
    
    ax2.set_title('端口类型任务分布', fontsize=12, fontweight='bold')
    
    plt.suptitle('资源维度分析：调度节点异常识别', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'resource_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f'资源分析图已保存: {output_path}')
    return output_path

if __name__ == '__main__':
    print('开始生成图表...')
    
    # 生成所有图表
    time_trend_path = create_time_trend_chart()
    host_comparison_path = create_host_comparison_chart()
    wait_distribution_path = create_wait_distribution_chart()
    resource_analysis_path = create_resource_analysis_chart()
    
    print('\n所有图表生成完成！')
    print(f'1. 时间趋势图: {time_trend_path}')
    print(f'2. 主机堆积对比图: {host_comparison_path}')
    print(f'3. 等待时长分布图: {wait_distribution_path}')
    print(f'4. 资源分析图: {resource_analysis_path}')
