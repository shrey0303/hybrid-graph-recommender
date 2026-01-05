"""
MLflow Experiment Tracker for ML Pipeline.

Wraps MLflow tracking API for recommendation model experiments.
Supports logging parameters, metrics, artifacts, and model registry.

Architecture:
    [Training Script] → [ExperimentTracker] → [MLflow Server]
                                                    ↓
                                            [Model Registry]
                                            [Artifact Store]
                                            [Metric History]
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ExperimentRun:
    """
    Represents a single experiment run with metrics and parameters.

    Attributes:
        run_id: Unique run identifier.
        experiment_name: Parent experiment name.
        parameters: Hyperparameters used.
        metrics: Training/evaluation metrics (name → list of values).
        artifacts: Paths to saved artifacts.
        status: Run status (running, completed, failed).
        start_time: Run start timestamp.
        end_time: Run end timestamp.
        tags: Custom tags for filtering.
    """
    run_id: str
    experiment_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    status: str = "running"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "metrics": {
                k: v[-1] if v else None for k, v in self.metrics.items()
            },
            "metric_history": self.metrics,
            "artifacts": self.artifacts,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (
                (self.end_time or time.time()) - self.start_time
            ),
            "tags": self.tags,
        }


class ExperimentTracker:
    """
    Track ML experiments with parameters, metrics, and artifacts.

    Provides a local tracking interface that mirrors MLflow's API.
    Can optionally integrate with a remote MLflow server.

    Supports:
    - Nested experiment organization
    - Parameter logging
    - Scalar metric tracking (with step history)
    - Artifact saving
    - Run comparison
    - Best run selection

    Example:
        >>> tracker = ExperimentTracker("recommendation_experiments")
        >>> run = tracker.start_run("gnn_baseline")
        >>> tracker.log_param("lr", 0.001)
        >>> tracker.log_metric("auc", 0.85, step=1)
        >>> tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str = "recommendation_system",
        tracking_dir: str = "mlruns",
        use_mlflow: bool = False,
    ) -> None:
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment group.
            tracking_dir: Directory for local tracking data.
            use_mlflow: Whether to connect to MLflow server.
        """
        self.experiment_name = experiment_name
        self.tracking_dir = tracking_dir
        self.use_mlflow = use_mlflow

        self.runs: Dict[str, ExperimentRun] = {}
        self.active_run: Optional[ExperimentRun] = None
        self._run_counter = 0

        os.makedirs(tracking_dir, exist_ok=True)

        if use_mlflow:
            self._init_mlflow()

        logger.info(
            f"ExperimentTracker initialized | "
            f"experiment='{experiment_name}', dir='{tracking_dir}'"
        )

    def _init_mlflow(self) -> None:
        """Initialize MLflow connection."""
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow integration enabled")
        except ImportError:
            logger.warning("MLflow not installed. Using local tracking only.")
            self.use_mlflow = False

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """
        Start a new experiment run.

        Args:
            run_name: Human-readable run name.
            tags: Custom tags for the run.

        Returns:
            The newly created ExperimentRun.
        """
        self._run_counter += 1
        run_id = f"run_{self._run_counter:04d}_{int(time.time())}"

        if run_name:
            run_id = f"{run_name}_{run_id}"

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            tags=tags or {},
        )

        self.runs[run_id] = run
        self.active_run = run

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.start_run(run_name=run_name)
                if tags:
                    mlflow.set_tags(tags)
            except Exception as e:
                logger.warning(f"MLflow start_run failed: {e}")

        logger.info(f"Started run: {run_id}")
        return run

    def log_param(self, key: str, value: Any) -> None:
        """Log a hyperparameter."""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.active_run.parameters[key] = value

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.log_param(key, value)
            except Exception:
                pass

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple hyperparameters at once."""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None
    ) -> None:
        """
        Log a scalar metric value.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number for tracking progress.
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        if key not in self.active_run.metrics:
            self.active_run.metrics[key] = []
        self.active_run.metrics[key].append(value)

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.log_metric(key, value, step=step)
            except Exception:
                pass

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, path: str) -> None:
        """Log a file artifact."""
        if self.active_run is None:
            raise RuntimeError("No active run.")
        self.active_run.artifacts.append(path)

    def end_run(self, status: str = "completed") -> ExperimentRun:
        """
        End the active run.

        Args:
            status: Final run status ('completed', 'failed').

        Returns:
            The completed ExperimentRun.
        """
        if self.active_run is None:
            raise RuntimeError("No active run to end.")

        self.active_run.status = status
        self.active_run.end_time = time.time()

        # Save run data locally
        self._save_run(self.active_run)

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.end_run(status=status)
            except Exception:
                pass

        run = self.active_run
        logger.info(
            f"Ended run {run.run_id} | Status: {status} | "
            f"Duration: {run.end_time - run.start_time:.1f}s"
        )

        self.active_run = None
        return run

    def _save_run(self, run: ExperimentRun) -> None:
        """Save run data to disk."""
        run_dir = os.path.join(self.tracking_dir, run.run_id)
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def get_best_run(self, metric: str, maximize: bool = True) -> Optional[ExperimentRun]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize.
            maximize: True for higher-is-better metrics.

        Returns:
            The best ExperimentRun, or None.
        """
        completed = [
            r for r in self.runs.values()
            if r.status == "completed" and metric in r.metrics
        ]

        if not completed:
            return None

        return max(
            completed,
            key=lambda r: r.metrics[metric][-1] * (1 if maximize else -1),
        )

    def compare_runs(
        self, run_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple runs side by side.

        Args:
            run_ids: Specific runs to compare. None = all runs.

        Returns:
            List of run summaries.
        """
        runs = (
            [self.runs[rid] for rid in run_ids if rid in self.runs]
            if run_ids
            else list(self.runs.values())
        )

        return [run.to_dict() for run in runs]

    def __repr__(self) -> str:
        return (
            f"ExperimentTracker("
            f"experiment='{self.experiment_name}', "
            f"runs={len(self.runs)}, "
            f"active={'yes' if self.active_run else 'no'})"
        )
