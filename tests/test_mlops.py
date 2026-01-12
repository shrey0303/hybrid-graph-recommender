"""
Unit tests for MLOps components.

Tests cover:
- ExperimentTracker run lifecycle
- Metric and parameter logging
- Best run selection
- MetricsCollector latency percentiles
- AlertManager rule checking
- MLPipelineDAG validation and topological sort
- Cycle detection
- Airflow DAG export
"""

import os
import time

import numpy as np
import pytest

from src.mlops.experiment_tracker import ExperimentTracker, ExperimentRun
from src.mlops.monitoring import MetricsCollector, AlertManager
from src.mlops.pipeline_dag import (
    MLPipelineDAG,
    PipelineTask,
    create_training_dag,
)


class TestExperimentTracker:
    """Tests for experiment tracker."""

    def test_start_and_end_run(self, tmp_path):
        """Test basic run lifecycle."""
        tracker = ExperimentTracker(
            tracking_dir=str(tmp_path / "runs")
        )
        run = tracker.start_run("test_run")
        assert run.status == "running"
        assert tracker.active_run is not None

        ended = tracker.end_run()
        assert ended.status == "completed"
        assert tracker.active_run is None

    def test_log_params(self, tmp_path):
        """Test parameter logging."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "runs"))
        tracker.start_run()
        tracker.log_param("lr", 0.001)
        tracker.log_param("epochs", 100)

        assert tracker.active_run.parameters["lr"] == 0.001
        assert tracker.active_run.parameters["epochs"] == 100

    def test_log_params_batch(self, tmp_path):
        """Test batch parameter logging."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "runs"))
        tracker.start_run()
        tracker.log_params({"lr": 0.01, "batch_size": 32})

        assert tracker.active_run.parameters["lr"] == 0.01

    def test_log_metrics(self, tmp_path):
        """Test metric logging with history."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "runs"))
        tracker.start_run()

        for i in range(5):
            tracker.log_metric("loss", 1.0 - i * 0.1, step=i)

        assert len(tracker.active_run.metrics["loss"]) == 5
        assert tracker.active_run.metrics["loss"][-1] == 0.6

    def test_log_without_run_raises(self, tmp_path):
        """Test logging without active run raises error."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "runs"))
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_param("lr", 0.01)

    def test_best_run(self, tmp_path):
        """Test best run selection."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "runs"))

        tracker.start_run("run_a")
        tracker.log_metric("auc", 0.75)
        tracker.end_run()

        tracker.start_run("run_b")
        tracker.log_metric("auc", 0.85)
        tracker.end_run()

        best = tracker.get_best_run("auc", maximize=True)
        assert best is not None
        assert best.metrics["auc"][-1] == 0.85

    def test_compare_runs(self, tmp_path):
        """Test run comparison."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "runs"))

        tracker.start_run("a")
        tracker.log_metric("loss", 0.5)
        tracker.end_run()

        tracker.start_run("b")
        tracker.log_metric("loss", 0.3)
        tracker.end_run()

        comparison = tracker.compare_runs()
        assert len(comparison) == 2

    def test_run_saved_to_disk(self, tmp_path):
        """Test that run data is persisted."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "runs"))
        run = tracker.start_run("persist_test")
        tracker.log_param("x", 42)
        tracker.end_run()

        run_dir = tmp_path / "runs" / run.run_id
        assert (run_dir / "run.json").exists()

    def test_repr(self, tmp_path):
        tracker = ExperimentTracker(tracking_dir=str(tmp_path / "r"))
        assert "ExperimentTracker" in repr(tracker)


class TestMetricsCollector:
    """Tests for metrics collection."""

    def test_record_latency(self):
        """Test latency recording."""
        collector = MetricsCollector()
        for _ in range(100):
            collector.record_latency("inference", np.random.uniform(10, 50))

        stats = collector.get_latency_stats("inference")
        assert stats["inference_p50"] > 0
        assert stats["inference_p99"] >= stats["inference_p50"]
        assert stats["inference_count"] == 100

    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector()
        collector.increment_counter("requests")
        collector.increment_counter("requests")
        collector.increment_counter("errors")

        report = collector.get_report()
        assert report["counters"]["requests"] == 2
        assert report["counters"]["errors"] == 1

    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector()
        collector.set_gauge("gpu_utilization", 0.75)

        report = collector.get_report()
        assert report["gauges"]["gpu_utilization"] == 0.75

    def test_prometheus_export(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        collector.increment_counter("requests")
        collector.record_latency("inference", 25.0)

        prom = collector.to_prometheus_format()
        assert "rec_requests_total" in prom
        assert "rec_inference" in prom

    def test_empty_latency_stats(self):
        """Test stats for non-existent metric."""
        collector = MetricsCollector()
        assert collector.get_latency_stats("unknown") == {}

    def test_reset(self):
        """Test metrics reset."""
        collector = MetricsCollector()
        collector.increment_counter("x")
        collector.reset()
        assert collector.get_report()["counters"] == {}

    def test_repr(self):
        collector = MetricsCollector()
        assert "MetricsCollector" in repr(collector)


class TestAlertManager:
    """Tests for alert management."""

    def test_add_rule(self):
        """Test adding alert rules."""
        alerts = AlertManager()
        alerts.add_rule("high_latency", "inference_p99", 100, "gt")
        assert "high_latency" in alerts.rules

    def test_alert_triggers(self):
        """Test that alert fires when threshold exceeded."""
        alerts = AlertManager(cooldown_seconds=0)
        alerts.add_rule("high_latency", "inference_p99", 100, "gt")

        triggered = alerts.check({"inference_p99": 150})
        assert len(triggered) == 1
        assert triggered[0]["rule"] == "high_latency"

    def test_alert_not_triggered(self):
        """Test that alert doesn't fire below threshold."""
        alerts = AlertManager()
        alerts.add_rule("high_latency", "inference_p99", 100, "gt")

        triggered = alerts.check({"inference_p99": 50})
        assert len(triggered) == 0

    def test_cooldown(self):
        """Test alert cooldown prevents spam."""
        alerts = AlertManager(cooldown_seconds=300)
        alerts.add_rule("test", "val", 10, "gt")

        alerts.check({"val": 20})
        triggered = alerts.check({"val": 20})
        # Second check should be suppressed by cooldown
        assert len(triggered) == 0

    def test_lt_comparison(self):
        """Test less-than comparison."""
        alerts = AlertManager(cooldown_seconds=0)
        alerts.add_rule("low_qps", "qps", 10, "lt", "critical")

        triggered = alerts.check({"qps": 5})
        assert len(triggered) == 1
        assert triggered[0]["severity"] == "critical"

    def test_repr(self):
        alerts = AlertManager()
        assert "AlertManager" in repr(alerts)


class TestMLPipelineDAG:
    """Tests for pipeline DAG."""

    def test_create_dag(self):
        """Test DAG creation."""
        dag = MLPipelineDAG("test_dag")
        assert dag.dag_id == "test_dag"
        assert len(dag.tasks) == 0

    def test_add_tasks(self):
        """Test adding tasks."""
        dag = MLPipelineDAG("test")
        dag.add_task(PipelineTask("load", "load_fn"))
        dag.add_task(PipelineTask("train", "train_fn", ["load"]))
        assert len(dag.tasks) == 2

    def test_validate_valid_dag(self):
        """Test validation of a valid DAG."""
        dag = MLPipelineDAG("test")
        dag.add_task(PipelineTask("a", "fn_a"))
        dag.add_task(PipelineTask("b", "fn_b", ["a"]))
        dag.add_task(PipelineTask("c", "fn_c", ["b"]))
        assert dag.validate()

    def test_validate_missing_dependency_raises(self):
        """Test that missing dependency fails validation."""
        dag = MLPipelineDAG("test")
        dag.add_task(PipelineTask("b", "fn_b", ["nonexistent"]))
        with pytest.raises(ValueError, match="non-existent"):
            dag.validate()

    def test_validate_cycle_raises(self):
        """Test that cyclic dependency fails validation."""
        dag = MLPipelineDAG("test")
        dag.add_task(PipelineTask("a", "fn_a", ["c"]))
        dag.add_task(PipelineTask("b", "fn_b", ["a"]))
        dag.add_task(PipelineTask("c", "fn_c", ["b"]))
        with pytest.raises(ValueError, match="Cycle"):
            dag.validate()

    def test_execution_order(self):
        """Test topological sort."""
        dag = MLPipelineDAG("test")
        dag.add_task(PipelineTask("a", "fn"))
        dag.add_task(PipelineTask("b", "fn", ["a"]))
        dag.add_task(PipelineTask("c", "fn", ["b"]))

        order = dag.get_execution_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_export_airflow_dag(self, tmp_path):
        """Test Airflow DAG file export."""
        dag = MLPipelineDAG("test_dag", description="Test pipeline")
        dag.add_task(PipelineTask("load", "load_fn"))
        dag.add_task(PipelineTask("train", "train_fn", ["load"]))

        path = dag.export_airflow_dag(str(tmp_path))
        assert os.path.isfile(path)

        content = open(path).read()
        assert "test_dag" in content
        assert "PythonOperator" in content
        assert "load >> train" in content

    def test_remove_task(self):
        """Test task removal with dependency cleanup."""
        dag = MLPipelineDAG("test")
        dag.add_task(PipelineTask("a", "fn"))
        dag.add_task(PipelineTask("b", "fn", ["a"]))
        dag.remove_task("a")
        assert "a" not in dag.tasks
        assert "a" not in dag.tasks["b"].dependencies

    def test_to_dict(self):
        """Test DAG serialization."""
        dag = MLPipelineDAG("test", schedule="0 * * * *")
        dag.add_task(PipelineTask("a", "fn"))
        d = dag.to_dict()
        assert d["dag_id"] == "test"
        assert "a" in d["tasks"]

    def test_create_training_dag(self):
        """Test pre-built training DAG."""
        dag = create_training_dag()
        assert dag.dag_id == "recommendation_training"
        assert len(dag.tasks) >= 7
        assert dag.validate()

    def test_repr(self):
        dag = MLPipelineDAG("test")
        assert "MLPipelineDAG" in repr(dag)
