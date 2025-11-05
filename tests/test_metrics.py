"""
Unit tests for recommendation metrics and reward analyzer.

Tests cover:
- NDCG@K, Precision@K, Recall@K, Hit Rate@K
- MAP, MRR computation
- Reward accuracy and margin statistics
- Model comparison
- RewardAnalyzer snapshot tracking and convergence analysis
"""

import numpy as np
import pytest

from src.evaluation.metrics import RecommendationMetrics
from src.evaluation.reward_analyzer import RewardAnalyzer


class TestNDCG:
    """Tests for NDCG@K metric."""

    def test_perfect_ranking(self):
        """NDCG should be 1.0 when relevant item is ranked first."""
        predicted = ["a", "b", "c"]
        relevant = ["a"]
        assert RecommendationMetrics.ndcg_at_k(predicted, relevant, k=3) == 1.0

    def test_worst_ranking(self):
        """NDCG should be < 1 when relevant item is ranked last."""
        predicted = ["b", "c", "a"]
        relevant = ["a"]
        score = RecommendationMetrics.ndcg_at_k(predicted, relevant, k=3)
        assert 0 < score < 1.0

    def test_no_relevant_items(self):
        """NDCG should be 0 when no relevant items in predictions."""
        predicted = ["b", "c", "d"]
        relevant = ["a"]
        assert RecommendationMetrics.ndcg_at_k(predicted, relevant, k=3) == 0.0

    def test_cutoff_k(self):
        """NDCG should respect K cutoff."""
        predicted = ["b", "c", "a"]
        relevant = ["a"]
        # K=2 should not find "a" (at position 3)
        assert RecommendationMetrics.ndcg_at_k(predicted, relevant, k=2) == 0.0


class TestPrecisionRecall:
    """Tests for Precision@K and Recall@K."""

    def test_precision_all_relevant(self):
        """Precision should be 1.0 when all predictions are relevant."""
        predicted = ["a", "b"]
        relevant = ["a", "b", "c"]
        assert RecommendationMetrics.precision_at_k(predicted, relevant, k=2) == 1.0

    def test_precision_none_relevant(self):
        """Precision should be 0 when no predictions are relevant."""
        predicted = ["d", "e"]
        relevant = ["a", "b"]
        assert RecommendationMetrics.precision_at_k(predicted, relevant, k=2) == 0.0

    def test_recall_all_found(self):
        """Recall should be 1.0 when all relevant items are in predictions."""
        predicted = ["a", "b", "c"]
        relevant = ["a", "b"]
        assert RecommendationMetrics.recall_at_k(predicted, relevant, k=3) == 1.0

    def test_recall_partial(self):
        """Recall should reflect fraction of relevant items found."""
        predicted = ["a", "d"]
        relevant = ["a", "b"]
        assert RecommendationMetrics.recall_at_k(predicted, relevant, k=2) == 0.5


class TestHitRateMRR:
    """Tests for Hit Rate@K and MRR."""

    def test_hit_rate_hit(self):
        """Hit rate should be 1 when relevant item appears in top-K."""
        predicted = ["b", "a", "c"]
        relevant = ["a"]
        assert RecommendationMetrics.hit_rate_at_k(predicted, relevant, k=3) == 1.0

    def test_hit_rate_miss(self):
        """Hit rate should be 0 when relevant item is not in top-K."""
        predicted = ["b", "c"]
        relevant = ["a"]
        assert RecommendationMetrics.hit_rate_at_k(predicted, relevant, k=2) == 0.0

    def test_mrr_first_position(self):
        """MRR should be 1.0 when relevant item is at position 1."""
        predicted = ["a", "b", "c"]
        relevant = ["a"]
        assert RecommendationMetrics.mrr(predicted, relevant) == 1.0

    def test_mrr_second_position(self):
        """MRR should be 0.5 when relevant item is at position 2."""
        predicted = ["b", "a", "c"]
        relevant = ["a"]
        assert RecommendationMetrics.mrr(predicted, relevant) == 0.5

    def test_mrr_not_found(self):
        """MRR should be 0 when relevant item is not in predictions."""
        predicted = ["b", "c"]
        relevant = ["a"]
        assert RecommendationMetrics.mrr(predicted, relevant) == 0.0


class TestAveragePrecision:
    """Tests for Average Precision."""

    def test_perfect_ap(self):
        """AP should be 1.0 when all relevant items are at the top."""
        predicted = ["a", "b", "c"]
        relevant = ["a", "b"]
        ap = RecommendationMetrics.average_precision(predicted, relevant)
        assert ap == 1.0

    def test_ap_with_gaps(self):
        """AP should account for gaps in relevant item positions."""
        predicted = ["a", "x", "b"]
        relevant = ["a", "b"]
        ap = RecommendationMetrics.average_precision(predicted, relevant)
        # AP = (1/1 + 2/3) / 2 = 0.833...
        assert abs(ap - 0.8333) < 0.01


class TestComputeAll:
    """Tests for batch metric computation."""

    def test_compute_all_returns_all_metrics(self):
        """Test that compute_all returns expected metric keys."""
        metrics = RecommendationMetrics()
        predictions = [["a", "b", "c"], ["d", "e", "f"]]
        ground_truth = [["a"], ["f"]]

        results = metrics.compute_all(predictions, ground_truth, k_values=[3])

        assert "ndcg@3" in results
        assert "precision@3" in results
        assert "recall@3" in results
        assert "hit_rate@3" in results
        assert "map" in results
        assert "mrr" in results

    def test_compute_all_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        metrics = RecommendationMetrics()
        with pytest.raises(ValueError, match="Length mismatch"):
            metrics.compute_all([["a"]], [["a"], ["b"]])


class TestRewardMetrics:
    """Tests for DPO reward metrics."""

    def test_reward_accuracy_perfect(self):
        """Reward accuracy should be 1.0 when chosen always wins."""
        chosen = np.array([0.9, 0.8, 0.7])
        rejected = np.array([0.1, 0.2, 0.3])
        assert RecommendationMetrics.reward_accuracy(chosen, rejected) == 1.0

    def test_reward_accuracy_zero(self):
        """Reward accuracy should be 0 when rejected always wins."""
        chosen = np.array([0.1, 0.2])
        rejected = np.array([0.9, 0.8])
        assert RecommendationMetrics.reward_accuracy(chosen, rejected) == 0.0

    def test_reward_margin(self):
        """Test reward margin statistics."""
        chosen = np.array([0.8, 0.7, 0.9])
        rejected = np.array([0.3, 0.4, 0.2])
        margin = RecommendationMetrics.reward_margin(chosen, rejected)

        assert margin["mean"] > 0
        assert margin["positive_ratio"] == 1.0
        assert "std" in margin


class TestModelComparison:
    """Tests for SFT vs DPO comparison."""

    def test_compare_models(self):
        """Test side-by-side model comparison."""
        sft_preds = [["b", "a"], ["f", "d"]]
        dpo_preds = [["a", "b"], ["d", "f"]]
        ground_truth = [["a"], ["d"]]

        result = RecommendationMetrics.compare_models(
            sft_preds, dpo_preds, ground_truth, k=2
        )

        assert "sft" in result
        assert "dpo" in result
        assert "improvement_pct" in result


class TestRewardAnalyzer:
    """Tests for RewardAnalyzer."""

    def test_add_snapshot(self):
        """Test adding a reward snapshot."""
        analyzer = RewardAnalyzer()
        analyzer.add_snapshot(
            step=100,
            chosen_rewards=[0.8, 0.7, 0.9],
            rejected_rewards=[0.3, 0.4, 0.2],
        )

        assert len(analyzer.history) == 1
        assert analyzer.history[0]["step"] == 100
        assert analyzer.history[0]["win_rate"] == 1.0

    def test_generate_report(self):
        """Test report generation with multiple snapshots."""
        analyzer = RewardAnalyzer()

        for step in range(0, 500, 100):
            margin = 0.1 + step * 0.001  # Improving margin
            analyzer.add_snapshot(
                step=step,
                chosen_rewards=[0.5 + margin, 0.6 + margin],
                rejected_rewards=[0.5 - margin, 0.6 - margin],
            )

        report = analyzer.generate_report()

        assert "overall" in report
        assert "convergence" in report
        assert report["overall"]["win_rate"] >= 0
        assert report["convergence"]["margin_improved"]

    def test_empty_report(self):
        """Test report with no data."""
        analyzer = RewardAnalyzer()
        report = analyzer.generate_report()
        assert report["status"] == "no_data"

    def test_summary_table(self):
        """Test formatted summary table."""
        analyzer = RewardAnalyzer()
        analyzer.add_snapshot(
            step=0, chosen_rewards=[0.5], rejected_rewards=[0.4]
        )
        table = analyzer.get_summary_table()
        assert "Step" in table
        assert "Win Rate" in table

    def test_repr(self):
        """Test string representation."""
        analyzer = RewardAnalyzer()
        assert "RewardAnalyzer" in repr(analyzer)
