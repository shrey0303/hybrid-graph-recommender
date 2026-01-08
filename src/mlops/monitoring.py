"""
Prometheus Monitoring for Recommendation Service.

Provides metrics collection, health monitoring, and alerting
configuration for the production recommendation pipeline.

Metrics tracked:
    - Inference latency (p50, p95, p99)
    - Request throughput (QPS)
    - Model prediction quality
    - Cache hit rates
    - System resource utilization
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class MetricsCollector:
    """
    Collect and aggregate service metrics for monitoring.

    Maintains sliding windows of observations for computing
    percentile-based latency metrics and throughput counters.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_latency("inference", 23.5)
        >>> collector.increment_counter("requests")
        >>> report = collector.get_report()
        >>> print(report["inference_p99"])
    """

    def __init__(self, window_size: int = 1000) -> None:
        """
        Initialize metrics collector.

        Args:
            window_size: Sliding window size for percentile computation.
        """
        self.window_size = window_size
        self._latencies: Dict[str, deque] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._start_time = time.time()

        logger.info(f"MetricsCollector initialized | window={window_size}")

    def record_latency(self, name: str, value_ms: float) -> None:
        """
        Record a latency observation.

        Args:
            name: Metric name (e.g., 'inference', 'graph_lookup').
            value_ms: Latency in milliseconds.
        """
        if name not in self._latencies:
            self._latencies[name] = deque(maxlen=self.window_size)
        self._latencies[name].append(value_ms)

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[name] = self._counters.get(name, 0) + amount

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric to a specific value."""
        self._gauges[name] = value

    def get_latency_stats(self, name: str) -> Dict[str, float]:
        """
        Get latency statistics for a metric.

        Returns:
            Dict with p50, p95, p99, mean, min, max.
        """
        if name not in self._latencies or not self._latencies[name]:
            return {}

        values = np.array(self._latencies[name])
        return {
            f"{name}_p50": float(np.percentile(values, 50)),
            f"{name}_p95": float(np.percentile(values, 95)),
            f"{name}_p99": float(np.percentile(values, 99)),
            f"{name}_mean": float(values.mean()),
            f"{name}_min": float(values.min()),
            f"{name}_max": float(values.max()),
            f"{name}_count": len(values),
        }

    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report.

        Returns:
            Dictionary with all metrics, latencies, and counters.
        """
        report: Dict[str, Any] = {
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

        # Latency stats
        for name in self._latencies:
            report.update(self.get_latency_stats(name))

        # Counters
        report["counters"] = dict(self._counters)

        # Gauges
        report["gauges"] = dict(self._gauges)

        # QPS computation
        uptime = max(time.time() - self._start_time, 1)
        total_requests = self._counters.get("requests", 0)
        report["qps"] = round(total_requests / uptime, 2)

        return report

    def to_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus exposition format.

        Returns:
            Prometheus-compatible metric string.
        """
        lines = []

        # Counters
        for name, value in self._counters.items():
            lines.append(f"# TYPE rec_{name}_total counter")
            lines.append(f"rec_{name}_total {value}")

        # Gauges
        for name, value in self._gauges.items():
            lines.append(f"# TYPE rec_{name} gauge")
            lines.append(f"rec_{name} {value}")

        # Latency histograms
        for name in self._latencies:
            stats = self.get_latency_stats(name)
            if stats:
                lines.append(f"# TYPE rec_{name}_ms summary")
                for k, v in stats.items():
                    lines.append(f"rec_{k} {v:.4f}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        self._latencies.clear()
        self._counters.clear()
        self._gauges.clear()
        self._start_time = time.time()

    def __repr__(self) -> str:
        return (
            f"MetricsCollector("
            f"latency_metrics={len(self._latencies)}, "
            f"counters={len(self._counters)})"
        )


class AlertManager:
    """
    Simple alert manager for metric threshold violations.

    Monitors metrics and triggers alerts when thresholds
    are exceeded. Supports configurable cooldown periods.

    Example:
        >>> alerts = AlertManager()
        >>> alerts.add_rule("latency_high", metric="inference_p99", threshold=100)
        >>> triggered = alerts.check(collector.get_report())
    """

    def __init__(self, cooldown_seconds: float = 300) -> None:
        """
        Initialize alert manager.

        Args:
            cooldown_seconds: Minimum time between repeated alerts.
        """
        self.cooldown_seconds = cooldown_seconds
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.last_triggered: Dict[str, float] = {}
        self.alert_history: List[Dict[str, Any]] = []

    def add_rule(
        self,
        name: str,
        metric: str,
        threshold: float,
        comparison: str = "gt",
        severity: str = "warning",
    ) -> None:
        """
        Add an alerting rule.

        Args:
            name: Rule name.
            metric: Metric key to monitor.
            threshold: Threshold value.
            comparison: 'gt' (greater than) or 'lt' (less than).
            severity: Alert severity (info, warning, critical).
        """
        self.rules[name] = {
            "metric": metric,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
        }

    def check(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all rules against current metrics.

        Args:
            metrics: Current metric values.

        Returns:
            List of triggered alerts.
        """
        triggered = []
        now = time.time()

        for name, rule in self.rules.items():
            metric_key = rule["metric"]
            value = metrics.get(metric_key)

            if value is None:
                continue

            # Check threshold
            violated = False
            if rule["comparison"] == "gt" and value > rule["threshold"]:
                violated = True
            elif rule["comparison"] == "lt" and value < rule["threshold"]:
                violated = True

            if violated:
                # Check cooldown
                last = self.last_triggered.get(name, 0)
                if now - last >= self.cooldown_seconds:
                    alert = {
                        "rule": name,
                        "metric": metric_key,
                        "value": value,
                        "threshold": rule["threshold"],
                        "severity": rule["severity"],
                        "timestamp": now,
                    }
                    triggered.append(alert)
                    self.alert_history.append(alert)
                    self.last_triggered[name] = now

                    logger.warning(
                        f"Alert [{rule['severity']}] {name}: "
                        f"{metric_key}={value:.2f} > {rule['threshold']}"
                    )

        return triggered

    def __repr__(self) -> str:
        return (
            f"AlertManager("
            f"rules={len(self.rules)}, "
            f"alerts_fired={len(self.alert_history)})"
        )
