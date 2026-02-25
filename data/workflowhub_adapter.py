# -*- coding: utf-8 -*-
"""
WfCommons/WorkflowHub 适配器
将 WfCommons 工作流转换为项目的 (tasks, resources, dependencies) 格式。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import random

try:
    from wfcommons import WorkflowGenerator
    from wfcommons.wfchef.recipes import (
        MontageRecipe,
        EpigenomicsRecipe,
        SeismologyRecipe,
        CyclesRecipe,
        BlastRecipe,
        BwaRecipe,
        GenomeRecipe,
        RnaseqRecipe,
        SoykbRecipe,
        SrasearchRecipe,
    )
except Exception:  # wfcommons 未安装
    WorkflowGenerator = None
    MontageRecipe = EpigenomicsRecipe = SeismologyRecipe = CyclesRecipe = None
    BlastRecipe = BwaRecipe = GenomeRecipe = RnaseqRecipe = None
    SoykbRecipe = SrasearchRecipe = None


RECIPE_MAP = {
    "montage": MontageRecipe,
    "epigenomics": EpigenomicsRecipe,
    "seismology": SeismologyRecipe,
    "cycles": CyclesRecipe,
    "blast": BlastRecipe,
    "bwa": BwaRecipe,
    "genome": GenomeRecipe,
    "rnaseq": RnaseqRecipe,
    "soykb": SoykbRecipe,
    "srasearch": SrasearchRecipe,
}


def wfcommons_available() -> bool:
    return WorkflowGenerator is not None


def get_available_recipes() -> List[str]:
    return [name for name, cls in RECIPE_MAP.items() if cls is not None]


def generate_workflow(
    recipe_name: str,
    num_tasks: int,
    runtime_factor: float = 1.0,
    input_file_size_factor: float = 1.0,
    output_file_size_factor: float = 1.0,
    seed: Optional[int] = None,
):
    """生成 WfCommons 工作流实例。"""
    if not wfcommons_available():
        raise ImportError(
            "wfcommons 未安装。请先安装 wfcommons 或使用回退工作流。"
        )

    recipe_key = recipe_name.strip().lower()
    recipe_cls = RECIPE_MAP.get(recipe_key)
    if recipe_cls is None:
        raise ValueError(
            f"未知配方: {recipe_name}. 可用: {', '.join(get_available_recipes())}"
        )

    if seed is not None:
        random.seed(seed)

    recipe = recipe_cls.from_num_tasks(
        num_tasks=num_tasks,
        runtime_factor=runtime_factor,
        input_file_size_factor=input_file_size_factor,
        output_file_size_factor=output_file_size_factor,
    )

    generator = WorkflowGenerator(recipe)
    return generator.build_workflow()


def workflow_to_project_format(
    workflow,
    num_resources: int = 6,
    resource_cpu_base: int = 4,
    resource_mem_base: int = 8,
) -> Tuple[List[Dict], List[Dict], List[Tuple[int, int]]]:
    """
    将 wfcommons Workflow 转成项目格式。

    Returns:
        tasks: [{id, name, duration, cpu_req, memory_req}]
        resources: [{id, name, cpu_capacity, memory_capacity, speed}]
        dependencies: [(pre, post)]
    """
    if workflow is None:
        raise ValueError("workflow 为空")

    tasks: List[Dict] = []
    node_to_id: Dict = {}

    # 构建任务列表
    for node in workflow.nodes():
        if str(node).upper() in ("SRC", "DST"):
            continue
        node_data = workflow.nodes[node]
        task_obj = node_data.get("task") if isinstance(node_data, dict) else None

        runtime = getattr(task_obj, "runtime", None)
        if runtime is None and isinstance(node_data, dict):
            runtime = node_data.get("runtime", None)
        duration = float(runtime) if runtime else 1.0

        cores = getattr(task_obj, "cores", None)
        if cores is None and isinstance(node_data, dict):
            cores = node_data.get("cores", None)
        cpu_req = int(cores) if cores else 1

        memory = getattr(task_obj, "memory", None)
        if memory is None and isinstance(node_data, dict):
            memory = node_data.get("memory", None)
        memory_req = int(memory) if memory else 1

        task_id = len(node_to_id)
        node_to_id[node] = task_id

        tasks.append(
            {
                "id": task_id,
                "name": str(getattr(task_obj, "name", node)),
                "duration": max(1.0, duration),
                "cpu_req": max(1, cpu_req),
                "memory_req": max(1, memory_req),
            }
        )

    # 构建依赖关系
    dependencies: List[Tuple[int, int]] = []
    for src, dst in workflow.edges():
        if src in node_to_id and dst in node_to_id:
            dependencies.append((node_to_id[src], node_to_id[dst]))

    # 构建资源列表
    resources: List[Dict] = []
    for i in range(num_resources):
        cpu_capacity = resource_cpu_base + (i % 4) * 2
        memory_capacity = resource_mem_base + (i % 4) * 4
        speed = 1.0 + (i % 3) * 0.2
        resources.append(
            {
                "id": i,
                "name": f"worker_{i}",
                "cpu_capacity": cpu_capacity,
                "memory_capacity": memory_capacity,
                "speed": speed,
            }
        )

    return tasks, resources, dependencies


def build_environment_from_recipe(
    recipe_name: str,
    num_tasks: int,
    num_resources: int = 6,
    runtime_factor: float = 1.0,
    input_file_size_factor: float = 1.0,
    output_file_size_factor: float = 1.0,
    seed: Optional[int] = None,
):
    """直接生成 EnhancedWorkflowSimulator 的输入数据。"""
    workflow = generate_workflow(
        recipe_name=recipe_name,
        num_tasks=num_tasks,
        runtime_factor=runtime_factor,
        input_file_size_factor=input_file_size_factor,
        output_file_size_factor=output_file_size_factor,
        seed=seed,
    )
    return workflow_to_project_format(workflow, num_resources=num_resources)
