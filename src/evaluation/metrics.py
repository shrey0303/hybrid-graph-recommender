"""
Recommendation-Specific Evaluation Metrics.

Provides standard information retrieval and recommendation metrics
for evaluating both SFT and DPO-aligned models. Supports per-query
and aggregate computation.

Metrics implemented:
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Precision@K
    - Recall@K
    - MAP (Mean Average Precision)
    - MRR (Mean Reciprocal Rank)
    - Hit Rate@K
    - DPO-specific: Reward accuracy, reward margin

References:
    - Järvelin & Kekäläinen, "Cumulated Gain-Based Evaluation of IR Techniques" (2002)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class RecommendationMetrics:
    """
    Compute recommendation system evaluation metrics.

    Supports both list-based metrics (ranking quality) and
    preference-based metrics (DPO alignment quality).

    Example:
        >>> metrics = RecommendationMetrics()
        >>> result = metrics.compute_all(
        ...     predictions=[["item_a", "item_b", "item_c"]],
        ...     ground_truth=[["item_b"]],
        ...     k=3,
        ... )
        >>> print(result)
    """

    @staticmethod
    def ndcg_at_k(
        predicted: List[str],
        relevant: List[str],
        k: int = 10,
    ) -> float:
        """
        Compute NDCG@K (Normalized Discounted Cumulative Gain).

        Measures ranking quality by penalizing relevant items appearing
        lower in the predicted ranking.

        Args:
            predicted: Ordered list of predicted items.
            relevant: Set of relevant (ground truth) items.
            k: Cutoff position.

        Returns:
            NDCG@K score in [0, 1].
        """
        predicted_k = predicted[:k]
        relevant_set = set(relevant)

        # DCG: sum of gains discounted by position
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # Ideal DCG: all relevant items at top
        ideal_k = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(
        predicted: List[str],
        relevant: List[str],
        k: int = 10,
    ) -> float:
        """
        Compute Precision@K.

        Fraction of top-K predictions that are relevant.

        Args:
            predicted: Ordered list of predicted items.
            relevant: Set of relevant items.
            k: Cutoff position.

        Returns:
            Precision@K in [0, 1].
        """
        predicted_k = set(predicted[:k])
        relevant_set = set(relevant)
        hits = len(predicted_k & relevant_set)
        return hits / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(
        predicted: List[str],
        relevant: List[str],
        k: int = 10,
    ) -> float:
        """
        Compute Recall@K.

        Fraction of relevant items that appear in top-K predictions.

        Args:
            predicted: Ordered list of predicted items.
            relevant: Set of relevant items.
            k: Cutoff position.

        Returns:
            Recall@K in [0, 1].
        """
        predicted_k = set(predicted[:k])
        relevant_set = set(relevant)

        if not relevant_set:
            return 0.0

        hits = len(predicted_k & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def hit_rate_at_k(
        predicted: List[str],
        relevant: List[str],
        k: int = 10,
    ) -> float:
        """
        Compute Hit Rate@K (binary: any relevant item in top-K?).

        Args:
            predicted: Ordered list of predicted items.
            relevant: Set of relevant items.
            k: Cutoff position.

        Returns:
            1.0 if any relevant item is in top-K, 0.0 otherwise.
        """
        predicted_k = set(predicted[:k])
        relevant_set = set(relevant)
        return 1.0 if predicted_k & relevant_set else 0.0

    @staticmethod
    def mrr(
        predicted: List[str],
        relevant: List[str],
    ) -> float:
        """
        Compute Reciprocal Rank (1/position of first relevant item).

        Args:
            predicted: Ordered list of predicted items.
            relevant: Set of relevant items.

        Returns:
            1/rank of first relevant item, or 0 if none found.
        """
        relevant_set = set(relevant)
        for i, item in enumerate(predicted):
            if item in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def average_precision(
        predicted: List[str],
        relevant: List[str],
    ) -> float:
        """
        Compute Average Precision for a single query.

        Args:
            predicted: Ordered list of predicted items.
            relevant: Set of relevant items.

        Returns:
            Average precision score.
        """
        relevant_set = set(relevant)
        hits = 0
        sum_precision = 0.0

        for i, item in enumerate(predicted):
            if item in relevant_set:
                hits += 1
                sum_precision += hits / (i + 1)

        return sum_precision / len(relevant_set) if relevant_set else 0.0

    def compute_all(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics across multiple queries.

        Args:
            predictions: List of predicted item rankings (one per query).
            ground_truth: List of relevant item sets (one per query).
            k_values: Cutoff values for @K metrics. Default: [1, 3, 5, 10, 20].

        Returns:
            Dictionary of aggregated metric scores.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(ground_truth)} ground truth"
            )

        results: Dict[str, float] = {}

        for k in k_values:
            ndcg_scores = []
            precision_scores = []
            recall_scores = []
            hit_rates = []

            for pred, gt in zip(predictions, ground_truth):
                ndcg_scores.append(self.ndcg_at_k(pred, gt, k))
                precision_scores.append(self.precision_at_k(pred, gt, k))
                recall_scores.append(self.recall_at_k(pred, gt, k))
                hit_rates.append(self.hit_rate_at_k(pred, gt, k))

            results[f"ndcg@{k}"] = float(np.mean(ndcg_scores))
            results[f"precision@{k}"] = float(np.mean(precision_scores))
            results[f"recall@{k}"] = float(np.mean(recall_scores))
            results[f"hit_rate@{k}"] = float(np.mean(hit_rates))

        # MAP and MRR (not @K dependent)
        ap_scores = [
            self.average_precision(pred, gt)
            for pred, gt in zip(predictions, ground_truth)
        ]
        mrr_scores = [
            self.mrr(pred, gt) for pred, gt in zip(predictions, ground_truth)
        ]

        results["map"] = float(np.mean(ap_scores))
        results["mrr"] = float(np.mean(mrr_scores))

        return results

    @staticmethod
    def reward_accuracy(
        chosen_scores: np.ndarray,
        rejected_scores: np.ndarray,
    ) -> float:
        """
        Compute reward accuracy (fraction where chosen > rejected).

        DPO-specific metric measuring alignment quality.

        Args:
            chosen_scores: Reward scores for chosen responses.
            rejected_scores: Reward scores for rejected responses.

        Returns:
            Fraction of pairs where chosen reward > rejected reward.
        """
        return float(np.mean(chosen_scores > rejected_scores))

    @staticmethod
    def reward_margin(
        chosen_scores: np.ndarray,
        rejected_scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute reward margin statistics.

        Args:
            chosen_scores: Reward scores for chosen responses.
            rejected_scores: Reward scores for rejected responses.

        Returns:
            Dictionary with margin mean, std, min, max.
        """
        margins = chosen_scores - rejected_scores
        return {
            "mean": float(np.mean(margins)),
            "std": float(np.std(margins)),
            "min": float(np.min(margins)),
            "max": float(np.max(margins)),
            "positive_ratio": float(np.mean(margins > 0)),
        }

    @staticmethod
    def compare_models(
        sft_predictions: List[List[str]],
        dpo_predictions: List[List[str]],
        ground_truth: List[List[str]],
        k: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare SFT vs DPO model performance side-by-side.

        Args:
            sft_predictions: SFT model predictions.
            dpo_predictions: DPO model predictions.
            ground_truth: Relevant item sets.
            k: Cutoff for @K metrics.

        Returns:
            Dictionary with 'sft', 'dpo', and 'improvement' metrics.
        """
        metrics = RecommendationMetrics()

        sft_results = metrics.compute_all(sft_predictions, ground_truth, [k])
        dpo_results = metrics.compute_all(dpo_predictions, ground_truth, [k])

        improvement = {}
        for key in sft_results:
            sft_val = sft_results[key]
            dpo_val = dpo_results[key]
            if sft_val > 0:
                improvement[key] = (dpo_val - sft_val) / sft_val * 100
            else:
                improvement[key] = float("inf") if dpo_val > 0 else 0.0

        return {
            "sft": sft_results,
            "dpo": dpo_results,
            "improvement_pct": improvement,
        }
