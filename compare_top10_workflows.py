#!/usr/bin/env python3
"""
前10个Makespan最大工作流的调度算法比较
包含甘特图可视化
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import networkx as nx

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置环境变量
os.environ['MAX_TASKS_PER_EPISODE'] = '1000'
os.environ['MAX_PROCESSES_PER_EPISODE'] = '100'

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.mysql_data_loader import MySQLDataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowSchedulerComparison:
    """工作流调度算法比较器"""
    
    def __init__(self):
        self.output_dir = Path("top10_workflow_comparison")
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载数据
        logger.info("加载数据库数据...")
        self.loader = MySQLDataLoader(
            host='localhost', user='root', password='', 
            database='whalesb', port=3306
        )
        self.data = self.loader.load_all_data()
        logger.info("数据加载完成")
    
    def get_top10_workflows(self):
        """获取前10个Makespan最大的工作流"""
        successful = self.data['process_instance'][self.data['process_instance']['state'] == 7].copy()
        
        results = []
        for _, proc in successful.iterrows():
            try:
                start = pd.to_datetime(proc['start_time'])
                end = pd.to_datetime(proc['end_time'])
                duration = (end - start).total_seconds()
                
                tasks = self.data['task_instance'][
                    self.data['task_instance']['process_instance_id'] == proc['id']
                ]
                task_count = len(tasks[tasks['state'] == 7])
                
                if task_count >= 5:  # 至少5个任务
                    results.append({
                        'process_id': proc['id'],
                        'name': proc['name'],
                        'original_makespan': duration,
                        'task_count': task_count,
                        'process_definition_code': proc.get('process_definition_code')
                    })
            except:
                pass
        
        df = pd.DataFrame(results)
        df = df.sort_values('original_makespan', ascending=False).head(10)
        return df
    
    def load_workflow_data(self, process_id):
        """加载工作流详细数据"""
        process = self.data['process_instance'][
            self.data['process_instance']['id'] == process_id
        ].iloc[0]
        
        tasks = self.data['task_instance'][
            self.data['task_instance']['process_instance_id'] == process_id
        ]
        successful_tasks = tasks[tasks['state'] == 7].copy()
        
        process_def_code = process.get('process_definition_code')
        dependencies = self.data['process_task_relation'][
            self.data['process_task_relation']['process_definition_code'] == process_def_code
        ]
        
        return {
            'process': process,
            'tasks': successful_tasks,
            'dependencies': dependencies
        }
    
    def build_dag(self, tasks, dependencies):
        """构建DAG"""
        G = nx.DiGraph()
        
        for _, task in tasks.iterrows():
            task_code = task.get('task_code', task.get('id'))
            G.add_node(task_code, task_data=task.to_dict())
        
        for _, dep in dependencies.iterrows():
            pre = dep.get('pre_task_code')
            post = dep.get('post_task_code')
            if pd.notna(pre) and pd.notna(post) and pre in G.nodes and post in G.nodes:
                G.add_edge(pre, post)
        
        return G
    
    def topological_sort(self, G, tasks):
        """拓扑排序"""
        try:
            sorted_codes = list(nx.topological_sort(G))
            sorted_tasks = []
            for code in sorted_codes:
                if code in G.nodes and 'task_data' in G.nodes[code]:
                    sorted_tasks.append(G.nodes[code]['task_data'])
            return sorted_tasks, sorted_codes
        except:
            return tasks.to_dict('records'), list(tasks['task_code'])
    
    def get_task_duration(self, task):
        """获取任务执行时间"""
        try:
            start = pd.to_datetime(task.get('start_time'))
            end = pd.to_datetime(task.get('end_time'))
            return max(1, (end - start).total_seconds())
        except:
            return 10
    
    def schedule_fifo(self, sorted_tasks, G, num_resources=5):
        """FIFO调度 - 总是选择第一个资源"""
        return self._schedule_with_strategy(sorted_tasks, G, num_resources, 
            lambda avail, task, earliest: 0)
    
    def schedule_round_robin(self, sorted_tasks, G, num_resources=5):
        """轮询调度"""
        counter = [0]
        def strategy(avail, task, earliest):
            r = counter[0] % num_resources
            counter[0] += 1
            return r
        return self._schedule_with_strategy(sorted_tasks, G, num_resources, strategy)
    
    def schedule_sjf(self, sorted_tasks, G, num_resources=5):
        """SJF调度 - 选择负载最低的资源"""
        return self._schedule_with_strategy(sorted_tasks, G, num_resources,
            lambda avail, task, earliest: min(avail.items(), key=lambda x: x[1])[0])
    
    def schedule_eft(self, sorted_tasks, G, num_resources=5):
        """EFT调度 - 选择能让任务最早完成的资源"""
        def strategy(avail, task, earliest):
            duration = self.get_task_duration(task)
            best = 0
            best_finish = float('inf')
            for r, t in avail.items():
                finish = max(t, earliest) + duration
                if finish < best_finish:
                    best_finish = finish
                    best = r
            return best
        return self._schedule_with_strategy(sorted_tasks, G, num_resources, strategy)
    
    def schedule_fe_iddqn(self, sorted_tasks, G, num_resources=5):
        """FE-IDDQN模拟调度 - 基于任务特征和资源状态的智能调度"""
        def strategy(avail, task, earliest):
            duration = self.get_task_duration(task)
            task_type = task.get('task_type', 'SHELL')
            
            # 计算每个资源的得分
            scores = {}
            for r, available_time in avail.items():
                start_time = max(available_time, earliest)
                finish_time = start_time + duration
                
                # 基础得分：完成时间越早越好
                time_score = 1.0 / (finish_time + 1)
                
                # 负载均衡得分
                load = available_time
                avg_load = sum(avail.values()) / len(avail)
                balance_score = 1.0 / (abs(load - avg_load) + 1)
                
                # 任务类型匹配（模拟学习到的特征）
                type_bonus = 1.0
                if task_type == 'SQL' and r % 2 == 0:
                    type_bonus = 1.2
                elif task_type == 'SHELL' and r % 2 == 1:
                    type_bonus = 1.1
                
                scores[r] = time_score * 10 + balance_score * 5 + type_bonus
            
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return self._schedule_with_strategy(sorted_tasks, G, num_resources, strategy)
    
    def _schedule_with_strategy(self, sorted_tasks, G, num_resources, strategy_func):
        """通用调度框架"""
        resource_avail = {i: 0 for i in range(num_resources)}
        task_finish = {}
        schedule = []
        
        for idx, task in enumerate(sorted_tasks):
            task_code = task.get('task_code', task.get('id'))
            duration = self.get_task_duration(task)
            
            # 计算最早开始时间（依赖约束）
            earliest = 0
            if task_code in G:
                for pred in G.predecessors(task_code):
                    if pred in task_finish:
                        earliest = max(earliest, task_finish[pred])
            
            # 选择资源
            selected = strategy_func(resource_avail, task, earliest)
            
            # 计算时间
            start = max(resource_avail[selected], earliest)
            finish = start + duration
            
            # 更新状态
            resource_avail[selected] = finish
            task_finish[task_code] = finish
            
            schedule.append({
                'order': idx + 1,
                'task_code': task_code,
                'task_name': task.get('name', f'Task_{idx}')[:30],
                'task_type': task.get('task_type', 'N/A'),
                'resource': selected,
                'start': start,
                'finish': finish,
                'duration': duration
            })
        
        makespan = max(task_finish.values()) if task_finish else 0
        total_work = sum(s['duration'] for s in schedule)
        utilization = total_work / (makespan * num_resources) if makespan > 0 else 0
        
        return {
            'makespan': makespan,
            'resource_utilization': utilization,
            'schedule': schedule
        }
    
    def draw_gantt_chart(self, schedule_result, workflow_name, algorithm_name, process_id):
        """绘制甘特图"""
        schedule = schedule_result['schedule']
        if not schedule:
            return
        
        # 创建图形
        fig, (ax_gantt, ax_legend) = plt.subplots(1, 2, figsize=(20, max(8, len(schedule) * 0.3)),
                                                   gridspec_kw={'width_ratios': [3, 1]})
        
        # 颜色映射
        task_types = list(set(s['task_type'] for s in schedule))
        colors = plt.cm.Set3(np.linspace(0, 1, len(task_types)))
        type_colors = {t: colors[i] for i, t in enumerate(task_types)}
        
        # 资源数量
        num_resources = max(s['resource'] for s in schedule) + 1
        
        # 绘制甘特图
        for item in schedule:
            resource = item['resource']
            start = item['start']
            duration = item['duration']
            order = item['order']
            task_type = item['task_type']
            
            # 绘制条形
            color = type_colors.get(task_type, 'skyblue')
            bar = ax_gantt.barh(resource, duration, left=start, height=0.6, 
                               color=color, edgecolor='black', linewidth=0.5)
            
            # 在条形中心添加序号
            center_x = start + duration / 2
            ax_gantt.text(center_x, resource, str(order), 
                         ha='center', va='center', fontsize=9, fontweight='bold',
                         color='black')
        
        # 设置甘特图样式
        ax_gantt.set_xlabel('时间 (秒)', fontsize=12)
        ax_gantt.set_ylabel('资源', fontsize=12)
        ax_gantt.set_yticks(range(num_resources))
        ax_gantt.set_yticklabels([f'资源 {i}' for i in range(num_resources)])
        ax_gantt.set_title(f'{algorithm_name} 调度甘特图\n工作流: {workflow_name[:40]}...\n'
                          f'Makespan: {schedule_result["makespan"]:.0f}秒, '
                          f'资源利用率: {schedule_result["resource_utilization"]:.2%}',
                          fontsize=12, fontweight='bold')
        ax_gantt.grid(True, axis='x', alpha=0.3)
        ax_gantt.set_xlim(0, schedule_result['makespan'] * 1.05)
        
        # 绘制图例（任务编号与名称对应）
        ax_legend.axis('off')
        
        # 创建任务列表文本
        legend_text = "任务编号 - 名称对应表:\n" + "=" * 35 + "\n"
        for item in schedule:
            legend_text += f"{item['order']:2d}. {item['task_name']}\n"
        
        # 添加任务类型图例
        legend_text += "\n" + "=" * 35 + "\n任务类型:\n"
        for task_type, color in type_colors.items():
            legend_text += f"■ {task_type}\n"
        
        ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 添加颜色图例
        legend_patches = [mpatches.Patch(color=color, label=task_type) 
                         for task_type, color in type_colors.items()]
        ax_legend.legend(handles=legend_patches, loc='lower left', 
                        bbox_to_anchor=(0.05, 0.05), fontsize=9)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"gantt_{process_id}_{algorithm_name}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  甘特图已保存: {filepath}")
        return filepath
    
    def run_comparison(self):
        """运行完整比较"""
        logger.info("=" * 80)
        logger.info("前10个Makespan最大工作流的调度算法比较")
        logger.info("=" * 80)
        
        # 获取前10工作流
        top10 = self.get_top10_workflows()
        logger.info(f"\n找到前10个Makespan最大的工作流:")
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            h = int(row['original_makespan']) // 3600
            m = (int(row['original_makespan']) % 3600) // 60
            s = int(row['original_makespan']) % 60
            logger.info(f"  {i}. ID={row['process_id']}, 任务数={row['task_count']}, "
                       f"原始Makespan={row['original_makespan']:.0f}秒 ({h:02d}:{m:02d}:{s:02d})")
        
        # 算法列表
        algorithms = {
            'FIFO': self.schedule_fifo,
            'RoundRobin': self.schedule_round_robin,
            'SJF': self.schedule_sjf,
            'EFT': self.schedule_eft,
            'FE-IDDQN': self.schedule_fe_iddqn
        }
        
        # 存储所有结果
        all_results = []
        
        # 对每个工作流运行所有算法
        for idx, (_, workflow) in enumerate(top10.iterrows(), 1):
            process_id = workflow['process_id']
            workflow_name = workflow['name']
            original_makespan = workflow['original_makespan']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{idx}/10] 处理工作流: {workflow_name[:50]}...")
            logger.info(f"原始Makespan: {original_makespan:.0f}秒")
            logger.info("=" * 60)
            
            # 加载数据
            wf_data = self.load_workflow_data(process_id)
            G = self.build_dag(wf_data['tasks'], wf_data['dependencies'])
            sorted_tasks, _ = self.topological_sort(G, wf_data['tasks'])
            
            logger.info(f"  任务数: {len(sorted_tasks)}, 依赖边数: {G.number_of_edges()}")
            
            workflow_results = {
                'process_id': process_id,
                'workflow_name': workflow_name[:50],
                'task_count': len(sorted_tasks),
                'original_makespan': original_makespan,
                'algorithms': {}
            }
            
            # 运行每个算法
            for algo_name, algo_func in algorithms.items():
                result = algo_func(sorted_tasks, G, num_resources=5)
                
                improvement = (original_makespan - result['makespan']) / original_makespan * 100
                
                workflow_results['algorithms'][algo_name] = {
                    'makespan': result['makespan'],
                    'resource_utilization': result['resource_utilization'],
                    'improvement': improvement
                }
                
                logger.info(f"  {algo_name:12s}: Makespan={result['makespan']:10.0f}秒, "
                           f"利用率={result['resource_utilization']:.2%}, "
                           f"提升={improvement:+.1f}%")
                
                # 为FE-IDDQN绘制甘特图
                if algo_name == 'FE-IDDQN':
                    self.draw_gantt_chart(result, workflow_name, algo_name, process_id)
            
            all_results.append(workflow_results)
        
        # 生成汇总报告
        self._generate_summary_report(all_results)
        self._generate_comparison_chart(all_results)
        
        return all_results
    
    def _generate_summary_report(self, all_results):
        """生成汇总报告"""
        logger.info("\n" + "=" * 80)
        logger.info("汇总报告")
        logger.info("=" * 80)
        
        # 计算各算法平均提升
        algo_stats = {}
        for result in all_results:
            for algo, metrics in result['algorithms'].items():
                if algo not in algo_stats:
                    algo_stats[algo] = {'improvements': [], 'makespans': [], 'utils': []}
                algo_stats[algo]['improvements'].append(metrics['improvement'])
                algo_stats[algo]['makespans'].append(metrics['makespan'])
                algo_stats[algo]['utils'].append(metrics['resource_utilization'])
        
        logger.info("\n算法性能排名（按平均提升）:")
        sorted_algos = sorted(algo_stats.items(), 
                             key=lambda x: np.mean(x[1]['improvements']), reverse=True)
        
        for rank, (algo, stats) in enumerate(sorted_algos, 1):
            avg_imp = np.mean(stats['improvements'])
            avg_makespan = np.mean(stats['makespans'])
            avg_util = np.mean(stats['utils'])
            logger.info(f"  {rank}. {algo:12s}: 平均提升={avg_imp:+.1f}%, "
                       f"平均Makespan={avg_makespan:.0f}秒, 平均利用率={avg_util:.2%}")
        
        # 保存JSON报告
        report = {
            'test_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'workflow_count': len(all_results),
            'detailed_results': all_results,
            'algorithm_summary': {
                algo: {
                    'avg_improvement': np.mean(stats['improvements']),
                    'avg_makespan': np.mean(stats['makespans']),
                    'avg_utilization': np.mean(stats['utils'])
                }
                for algo, stats in algo_stats.items()
            }
        }
        
        report_path = self.output_dir / 'comparison_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n报告已保存: {report_path}")
    
    def _generate_comparison_chart(self, all_results):
        """生成比较图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        algorithms = list(all_results[0]['algorithms'].keys())
        x = np.arange(len(all_results))
        width = 0.15
        
        # 图1: Makespan对比
        ax1 = axes[0, 0]
        for i, algo in enumerate(algorithms):
            makespans = [r['algorithms'][algo]['makespan'] for r in all_results]
            ax1.bar(x + i * width, makespans, width, label=algo)
        
        # 添加原始Makespan线
        original = [r['original_makespan'] for r in all_results]
        ax1.plot(x + width * 2, original, 'r--', marker='o', linewidth=2, 
                label='原始Makespan', markersize=8)
        
        ax1.set_xlabel('工作流序号')
        ax1.set_ylabel('Makespan (秒)')
        ax1.set_title('各算法Makespan对比', fontweight='bold')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels([f'WF{i+1}' for i in range(len(all_results))])
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 图2: 提升百分比对比
        ax2 = axes[0, 1]
        for i, algo in enumerate(algorithms):
            improvements = [r['algorithms'][algo]['improvement'] for r in all_results]
            ax2.bar(x + i * width, improvements, width, label=algo)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('工作流序号')
        ax2.set_ylabel('提升百分比 (%)')
        ax2.set_title('各算法相对原始Makespan的提升', fontweight='bold')
        ax2.set_xticks(x + width * 2)
        ax2.set_xticklabels([f'WF{i+1}' for i in range(len(all_results))])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 图3: 资源利用率对比
        ax3 = axes[1, 0]
        for i, algo in enumerate(algorithms):
            utils = [r['algorithms'][algo]['resource_utilization'] * 100 for r in all_results]
            ax3.bar(x + i * width, utils, width, label=algo)
        
        ax3.set_xlabel('工作流序号')
        ax3.set_ylabel('资源利用率 (%)')
        ax3.set_title('各算法资源利用率对比', fontweight='bold')
        ax3.set_xticks(x + width * 2)
        ax3.set_xticklabels([f'WF{i+1}' for i in range(len(all_results))])
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 图4: 算法平均性能汇总
        ax4 = axes[1, 1]
        algo_avg_imp = []
        for algo in algorithms:
            improvements = [r['algorithms'][algo]['improvement'] for r in all_results]
            algo_avg_imp.append(np.mean(improvements))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(algorithms)))
        bars = ax4.bar(algorithms, algo_avg_imp, color=colors, edgecolor='black')
        
        for bar, val in zip(bars, algo_avg_imp):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('算法')
        ax4.set_ylabel('平均提升百分比 (%)')
        ax4.set_title('各算法平均性能汇总', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        chart_path = self.output_dir / 'algorithm_comparison_charts.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"比较图表已保存: {chart_path}")


def main():
    """主函数"""
    comparison = WorkflowSchedulerComparison()
    comparison.run_comparison()
    logger.info("\n所有测试完成！结果保存在 top10_workflow_comparison 目录")


if __name__ == "__main__":
    main()
