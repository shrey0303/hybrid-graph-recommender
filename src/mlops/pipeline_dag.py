"""
Airflow DAG Definitions for ML Pipeline Orchestration.

Defines DAGs for:
1. Model training pipeline (daily)
2. Data preprocessing and feature engineering
3. Model evaluation and promotion
4. Monitoring and alerting

These are DAG definitions that Airflow would import.
They also serve as documentation of the production pipeline.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class PipelineTask:
    """
    A task in the ML pipeline DAG.

    Attributes:
        task_id: Unique task identifier.
        callable_name: Python function to execute.
        dependencies: List of task_ids this task depends on.
        retries: Number of retry attempts.
        timeout_minutes: Task timeout.
        params: Task-specific parameters.
    """
    task_id: str
    callable_name: str
    dependencies: List[str] = field(default_factory=list)
    retries: int = 2
    timeout_minutes: int = 60
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "callable": self.callable_name,
            "dependencies": self.dependencies,
            "retries": self.retries,
            "timeout_minutes": self.timeout_minutes,
            "params": self.params,
        }


class MLPipelineDAG:
    """
    Define and manage ML pipeline Directed Acyclic Graphs.

    Provides a framework for defining multi-stage ML pipelines
    that can be exported as Airflow DAG definitions.

    Pipelines:
    1. Training: Data → Feature Eng → Train → Evaluate → Promote
    2. Serving: Load Model → Warm Cache → Health Check → Deploy
    3. Monitoring: Collect Metrics → Check Alerts → Report

    Example:
        >>> dag = MLPipelineDAG("recommendation_training")
        >>> dag.add_task(PipelineTask("load_data", "load_dataset"))
        >>> dag.add_task(PipelineTask("train", "train_model", ["load_data"]))
        >>> dag.validate()
        >>> dag.export_airflow_dag("dags/")
    """

    def __init__(
        self,
        dag_id: str,
        schedule: str = "0 2 * * *",  # Daily at 2 AM
        description: str = "",
        max_active_runs: int = 1,
    ) -> None:
        """
        Initialize pipeline DAG.

        Args:
            dag_id: Unique DAG identifier.
            schedule: Cron schedule expression.
            description: DAG description.
            max_active_runs: Max concurrent DAG runs.
        """
        self.dag_id = dag_id
        self.schedule = schedule
        self.description = description
        self.max_active_runs = max_active_runs

        self.tasks: Dict[str, PipelineTask] = {}

        logger.info(f"MLPipelineDAG created: {dag_id}")

    def add_task(self, task: PipelineTask) -> None:
        """Add a task to the DAG."""
        self.tasks[task.task_id] = task

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the DAG."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            # Remove from dependencies
            for task in self.tasks.values():
                if task_id in task.dependencies:
                    task.dependencies.remove(task_id)

    def validate(self) -> bool:
        """
        Validate DAG structure (no cycles, valid dependencies).

        Returns:
            True if DAG is valid.

        Raises:
            ValueError: If DAG has cycles or invalid dependencies.
        """
        # Check for missing dependencies
        all_ids = set(self.tasks.keys())
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in all_ids:
                    raise ValueError(
                        f"Task '{task.task_id}' depends on "
                        f"non-existent task '{dep}'"
                    )

        # Check for cycles using topological sort
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            for dep in self.tasks[task_id].dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.discard(task_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError(f"Cycle detected in DAG '{self.dag_id}'")

        logger.info(f"DAG '{self.dag_id}' validated: {len(self.tasks)} tasks")
        return True

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order.

        Returns:
            List of task_ids in execution order.
        """
        visited = set()
        order = []

        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep in self.tasks[task_id].dependencies:
                visit(dep)
            order.append(task_id)

        for task_id in self.tasks:
            visit(task_id)

        return order

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary."""
        return {
            "dag_id": self.dag_id,
            "schedule": self.schedule,
            "description": self.description,
            "max_active_runs": self.max_active_runs,
            "tasks": {
                tid: task.to_dict() for tid, task in self.tasks.items()
            },
            "execution_order": self.get_execution_order(),
        }

    def export_airflow_dag(self, output_dir: str) -> str:
        """
        Export as an Airflow Python DAG file.

        Args:
            output_dir: Directory to write the DAG file.

        Returns:
            Path to the generated DAG file.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, f"{self.dag_id}.py")
        execution_order = self.get_execution_order()

        lines = [
            '"""Auto-generated Airflow DAG."""',
            "",
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "from datetime import datetime, timedelta",
            "",
            "default_args = {",
            f"    'retries': 2,",
            f"    'retry_delay': timedelta(minutes=5),",
            "}",
            "",
            f"with DAG(",
            f"    dag_id='{self.dag_id}',",
            f"    schedule='{self.schedule}',",
            f"    default_args=default_args,",
            f"    start_date=datetime(2025, 1, 1),",
            f"    catchup=False,",
            f"    max_active_runs={self.max_active_runs},",
            f"    description='{self.description}',",
            ") as dag:",
            "",
        ]

        # Generate tasks
        for task_id in execution_order:
            task = self.tasks[task_id]
            lines.append(
                f"    {task_id} = PythonOperator("
            )
            lines.append(f"        task_id='{task_id}',")
            lines.append(f"        python_callable={task.callable_name},")
            if task.params:
                lines.append(f"        op_kwargs={task.params},")
            lines.append(f"        retries={task.retries},")
            lines.append(
                f"        execution_timeout=timedelta(minutes={task.timeout_minutes}),"
            )
            lines.append("    )")
            lines.append("")

        # Generate dependency chains
        lines.append("    # Dependencies")
        for task in self.tasks.values():
            for dep in task.dependencies:
                lines.append(f"    {dep} >> {task.task_id}")

        content = "\n".join(lines) + "\n"

        with open(filepath, "w") as f:
            f.write(content)

        logger.info(f"Exported Airflow DAG to {filepath}")
        return filepath

    def __repr__(self) -> str:
        return (
            f"MLPipelineDAG("
            f"id='{self.dag_id}', "
            f"tasks={len(self.tasks)}, "
            f"schedule='{self.schedule}')"
        )


def create_training_dag() -> MLPipelineDAG:
    """
    Create the standard recommendation model training DAG.

    Pipeline:
    load_data → build_graph → train_gnn → train_dpo
                                    ↓          ↓
                               evaluate_gnn  evaluate_dpo
                                    ↓          ↓
                               promote_model ←─┘
    """
    dag = MLPipelineDAG(
        dag_id="recommendation_training",
        schedule="0 2 * * *",
        description="Daily recommendation model training pipeline",
    )

    dag.add_task(PipelineTask(
        "load_data", "load_dataset",
        params={"data_dir": "./", "format": "excel"},
    ))
    dag.add_task(PipelineTask(
        "build_graph", "build_interaction_graph",
        dependencies=["load_data"],
        params={"embedding_dim": 128},
    ))
    dag.add_task(PipelineTask(
        "train_gnn", "train_graphsage",
        dependencies=["build_graph"],
        timeout_minutes=120,
        params={"epochs": 100, "lr": 0.001},
    ))
    dag.add_task(PipelineTask(
        "generate_preferences", "generate_dpo_data",
        dependencies=["load_data"],
    ))
    dag.add_task(PipelineTask(
        "train_dpo", "train_dpo_model",
        dependencies=["generate_preferences"],
        timeout_minutes=180,
        params={"beta": 0.1, "epochs": 3},
    ))
    dag.add_task(PipelineTask(
        "evaluate", "evaluate_models",
        dependencies=["train_gnn", "train_dpo"],
    ))
    dag.add_task(PipelineTask(
        "promote_model", "promote_best_model",
        dependencies=["evaluate"],
        params={"metric": "ndcg@10", "threshold": 0.7},
    ))

    return dag
