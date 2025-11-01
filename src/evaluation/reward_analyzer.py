"""
Reward Analyzer for DPO Training Diagnostics.

Tracks and visualizes reward distributions throughout DPO training.
Provides insight into how well the model distinguishes between chosen
(correct) and rejected (incorrect) recommendations.

Key analyses:
    - Reward distribution comparison (chosen vs rejected)
    - Reward margin convergence over training
    - Win rate tracking (% chosen > rejected)
    - KL divergence estimation from reference model
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class RewardAnalyzer:
    """
    Analyze reward signals from DPO-aligned recommendation models.

    Tracks reward distributions, margins, and alignment quality
    metrics throughout and after training.

    Attributes:
        history: List of reward snapshots taken at different steps.

    Example:
        >>> analyzer = RewardAnalyzer()
        >>> analyzer.add_snapshot(step=100, chosen=[0.8, 0.7], rejected=[0.3, 0.2])
        >>> report = analyzer.generate_report()
        >>> print(report["win_rate"])
    """

    def __init__(self) -> None:
        """Initialize reward analyzer."""
        self.history: List[Dict[str, Any]] = []
        self._all_chosen: List[float] = []
        self._all_rejected: List[float] = []

    def add_snapshot(
        self,
        step: int,
        chosen_rewards: List[float],
        rejected_rewards: List[float],
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Record a reward snapshot during training.

        Args:
            step: Training step / epoch number.
            chosen_rewards: Batch of chosen response rewards.
            rejected_rewards: Batch of rejected response rewards.
            metadata: Optional additional info (loss, lr, etc.).
        """
        chosen = np.array(chosen_rewards)
        rejected = np.array(rejected_rewards)
        margins = chosen - rejected

        snapshot = {
            "step": step,
            "chosen_mean": float(chosen.mean()),
            "chosen_std": float(chosen.std()),
            "rejected_mean": float(rejected.mean()),
            "rejected_std": float(rejected.std()),
            "margin_mean": float(margins.mean()),
            "margin_std": float(margins.std()),
            "win_rate": float(np.mean(margins > 0)),
            "num_samples": len(chosen_rewards),
        }

        if metadata:
            snapshot.update(metadata)

        self.history.append(snapshot)
        self._all_chosen.extend(chosen_rewards)
        self._all_rejected.extend(rejected_rewards)

        logger.debug(
            f"Snapshot @ step {step} | "
            f"Margin: {snapshot['margin_mean']:.4f} ± {snapshot['margin_std']:.4f} | "
            f"Win rate: {snapshot['win_rate']:.4f}"
        )

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive reward analysis report.

        Returns:
            Dictionary with:
            - overall: Aggregate statistics across all snapshots
            - convergence: Whether margins are improving over time
            - distribution: Reward distribution analysis
            - history: Full snapshot history
        """
        if not self.history:
            return {"status": "no_data"}

        chosen = np.array(self._all_chosen)
        rejected = np.array(self._all_rejected)
        margins = chosen - rejected

        report = {
            "overall": {
                "total_samples": len(chosen),
                "total_snapshots": len(self.history),
                "chosen_reward_mean": float(chosen.mean()),
                "rejected_reward_mean": float(rejected.mean()),
                "margin_mean": float(margins.mean()),
                "margin_std": float(margins.std()),
                "win_rate": float(np.mean(margins > 0)),
                "strong_win_rate": float(np.mean(margins > 0.1)),
            },
            "convergence": self._analyze_convergence(),
            "distribution": {
                "chosen_quartiles": {
                    "q25": float(np.percentile(chosen, 25)),
                    "q50": float(np.percentile(chosen, 50)),
                    "q75": float(np.percentile(chosen, 75)),
                },
                "rejected_quartiles": {
                    "q25": float(np.percentile(rejected, 25)),
                    "q50": float(np.percentile(rejected, 50)),
                    "q75": float(np.percentile(rejected, 75)),
                },
                "margin_quartiles": {
                    "q25": float(np.percentile(margins, 25)),
                    "q50": float(np.percentile(margins, 50)),
                    "q75": float(np.percentile(margins, 75)),
                },
            },
            "history": self.history,
        }

        logger.info(
            f"Reward report generated | "
            f"Win rate: {report['overall']['win_rate']:.4f} | "
            f"Margin: {report['overall']['margin_mean']:.4f}"
        )

        return report

    def _analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze if reward margins are improving over training.

        Checks whether the margin trend is positive (model is learning
        to prefer chosen over rejected) by fitting a simple linear trend.

        Returns:
            Convergence analysis dictionary.
        """
        if len(self.history) < 2:
            return {"converged": False, "reason": "insufficient_data"}

        steps = [s["step"] for s in self.history]
        margins = [s["margin_mean"] for s in self.history]
        win_rates = [s["win_rate"] for s in self.history]

        # Simple linear trend
        if len(steps) >= 3:
            margin_trend = np.polyfit(range(len(margins)), margins, 1)[0]
            win_rate_trend = np.polyfit(range(len(win_rates)), win_rates, 1)[0]
        else:
            margin_trend = margins[-1] - margins[0]
            win_rate_trend = win_rates[-1] - win_rates[0]

        # Check early vs late performance
        early_margin = np.mean(margins[:max(1, len(margins) // 3)])
        late_margin = np.mean(margins[-max(1, len(margins) // 3):])

        return {
            "margin_trend": float(margin_trend),
            "win_rate_trend": float(win_rate_trend),
            "early_margin_mean": float(early_margin),
            "late_margin_mean": float(late_margin),
            "margin_improved": late_margin > early_margin,
            "final_win_rate": float(win_rates[-1]),
            "convergence_quality": (
                "good" if win_rates[-1] > 0.7
                else "moderate" if win_rates[-1] > 0.5
                else "poor"
            ),
        }

    def get_summary_table(self) -> str:
        """
        Generate a formatted summary table of snapshots.

        Returns:
            Formatted string table.
        """
        if not self.history:
            return "No snapshots recorded."

        header = f"{'Step':>8} | {'Chosen':>8} | {'Rejected':>8} | {'Margin':>8} | {'Win Rate':>8}"
        divider = "-" * len(header)
        rows = [header, divider]

        for snap in self.history:
            rows.append(
                f"{snap['step']:>8d} | "
                f"{snap['chosen_mean']:>8.4f} | "
                f"{snap['rejected_mean']:>8.4f} | "
                f"{snap['margin_mean']:>8.4f} | "
                f"{snap['win_rate']:>8.4f}"
            )

        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"RewardAnalyzer("
            f"snapshots={len(self.history)}, "
            f"samples={len(self._all_chosen)})"
        )
