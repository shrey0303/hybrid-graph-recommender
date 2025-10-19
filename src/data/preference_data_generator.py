"""
Preference Data Generator for DPO Training.

Creates (prompt, chosen, rejected) triplets from the Amazon recommendation
dataset for Direct Preference Optimization. Leverages existing ground truth
labels and model predictions to construct high-quality preference pairs.

Data Sources:
    - final_dataset.xlsx: User purchase histories → prompts
    - ground_truth.csv: Correct next-item predictions → chosen responses
    - final_generated_output.csv: Model predictions → rejected responses
      (when they differ from ground truth)

DPO Training Format:
    Each sample is a triplet:
    - prompt: "User has bought X, Y, Z. What should they buy next?"
    - chosen: Ground truth item (correct recommendation)
    - rejected: Wrong model prediction (incorrect recommendation)

References:
    - Rafailov et al., "Direct Preference Optimization" (2023)
    - https://arxiv.org/abs/2305.18290
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class PreferenceDataGenerator:
    """
    Generate preference pairs for DPO training from recommendation data.

    Creates (prompt, chosen, rejected) triplets where:
    - chosen = ground truth next-item recommendation
    - rejected = incorrect model prediction

    Supports multiple negative mining strategies:
    - Random: Random incorrect item from the catalog
    - Model-based: Use actual model mistakes as negatives
    - Hard: Items similar to the correct answer but wrong

    Attributes:
        data_dir: Root directory containing data files.
        prompts: List of recommendation prompts.
        chosen_items: Ground truth correct items.
        rejected_items: Incorrect model predictions.
        preference_pairs: Generated (prompt, chosen, rejected) triplets.

    Example:
        >>> generator = PreferenceDataGenerator("./")
        >>> generator.load_data()
        >>> pairs = generator.generate_pairs(strategy="model_based")
        >>> dataset = generator.to_hf_dataset()
        >>> print(f"Generated {len(pairs)} preference pairs")
    """

    def __init__(self, data_dir: str) -> None:
        """
        Initialize preference data generator.

        Args:
            data_dir: Path to directory containing data files.

        Raises:
            FileNotFoundError: If data directory doesn't exist.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.data_dir = data_dir
        self.prompts: List[str] = []
        self.chosen_items: List[str] = []
        self.model_predictions: List[str] = []
        self.all_items: List[str] = []
        self.preference_pairs: List[Dict[str, str]] = []
        self._is_loaded = False

        logger.info(f"PreferenceDataGenerator initialized with data_dir={data_dir}")

    def load_data(self) -> None:
        """
        Load prompts, ground truth, and model predictions from disk.

        Reads three data sources and aligns them by index for
        preference pair construction.

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If data files are empty or have mismatched lengths.
        """
        # Load ground truth items
        gt_path = os.path.join(self.data_dir, "ground_truth.csv")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        gt_df = pd.read_csv(gt_path, header=None, names=["item"])
        self.chosen_items = gt_df["item"].astype(str).str.strip().tolist()

        # Load model predictions
        pred_path = os.path.join(self.data_dir, "final_generated_output.csv")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")

        pred_df = pd.read_csv(pred_path, header=None, names=["item"])
        self.model_predictions = pred_df["item"].astype(str).str.strip().tolist()

        # Load prompts from Excel dataset
        xlsx_path = os.path.join(self.data_dir, "final_dataset.xlsx")
        if os.path.isfile(xlsx_path):
            try:
                xlsx_df = pd.read_excel(xlsx_path, engine="openpyxl")
                if "prompts" in xlsx_df.columns:
                    self.prompts = xlsx_df["prompts"].astype(str).str.strip().tolist()
                else:
                    logger.warning("No 'prompts' column in Excel. Generating synthetic prompts.")
                    self.prompts = self._generate_synthetic_prompts()
            except Exception as e:
                logger.warning(f"Could not read Excel file: {e}. Using synthetic prompts.")
                self.prompts = self._generate_synthetic_prompts()
        else:
            logger.info("Excel file not found. Generating synthetic prompts.")
            self.prompts = self._generate_synthetic_prompts()

        # Build catalog of all unique items
        self.all_items = list(set(self.chosen_items + self.model_predictions))

        # Align lengths (use the minimum common length)
        min_len = min(len(self.prompts), len(self.chosen_items), len(self.model_predictions))

        if min_len == 0:
            raise ValueError("No data found. Check that data files are not empty.")

        self.prompts = self.prompts[:min_len]
        self.chosen_items = self.chosen_items[:min_len]
        self.model_predictions = self.model_predictions[:min_len]

        self._is_loaded = True

        logger.info(
            f"Data loaded | Prompts: {len(self.prompts)} | "
            f"Ground truth items: {len(self.chosen_items)} | "
            f"Model predictions: {len(self.model_predictions)} | "
            f"Unique items in catalog: {len(self.all_items)}"
        )

    def _generate_synthetic_prompts(self) -> List[str]:
        """
        Generate synthetic prompts when the Excel file is unavailable.

        Creates prompts from ground truth items by simulating purchase histories.

        Returns:
            List of synthetic prompt strings.
        """
        prompts = []
        for i, item in enumerate(self.chosen_items):
            # Create a simple prompt template
            prompt = (
                f"A user has been shopping for grocery and household items. "
                f"Based on their purchase history, predict the next item "
                f"they would like to buy. The recommendation should be a "
                f"specific product name."
            )
            prompts.append(prompt)
        return prompts

    def generate_pairs(
        self,
        strategy: str = "model_based",
        num_negatives_per_positive: int = 1,
        random_state: int = 42,
    ) -> List[Dict[str, str]]:
        """
        Generate preference pairs using the specified strategy.

        Strategies:
        - "model_based": Use actual model mistakes as rejected items.
          Only creates pairs where model prediction ≠ ground truth.
        - "random": Sample random incorrect items as rejected.
        - "mixed": Combination of model-based and random negatives.

        Args:
            strategy: Negative sampling strategy.
            num_negatives_per_positive: Negatives per positive (for random/mixed).
            random_state: Random seed for reproducibility.

        Returns:
            List of dicts with keys: prompt, chosen, rejected.

        Raises:
            RuntimeError: If data hasn't been loaded yet.
            ValueError: If strategy is unknown.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        valid_strategies = {"model_based", "random", "mixed"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {valid_strategies}"
            )

        rng = random.Random(random_state)
        self.preference_pairs = []

        if strategy == "model_based":
            self._generate_model_based_pairs()
        elif strategy == "random":
            self._generate_random_pairs(rng, num_negatives_per_positive)
        elif strategy == "mixed":
            self._generate_model_based_pairs()
            self._generate_random_pairs(
                rng, num_negatives_per_positive, append=True
            )

        # Shuffle pairs
        rng.shuffle(self.preference_pairs)

        logger.info(
            f"Generated {len(self.preference_pairs)} preference pairs "
            f"using strategy='{strategy}'"
        )

        return self.preference_pairs

    def _generate_model_based_pairs(self) -> None:
        """
        Create pairs where the model's prediction was wrong.

        For each sample where model_prediction ≠ ground_truth,
        create a pair with chosen=ground_truth, rejected=model_prediction.
        """
        pairs_created = 0
        pairs_skipped = 0

        for i in range(len(self.prompts)):
            chosen = self.chosen_items[i]
            predicted = self.model_predictions[i]

            # Only create pair if the model got it wrong
            if self._items_differ(chosen, predicted):
                self.preference_pairs.append({
                    "prompt": self._format_dpo_prompt(self.prompts[i]),
                    "chosen": self._format_response(chosen),
                    "rejected": self._format_response(predicted),
                })
                pairs_created += 1
            else:
                pairs_skipped += 1

        logger.info(
            f"Model-based pairs: {pairs_created} created, "
            f"{pairs_skipped} skipped (model was correct)"
        )

    def _generate_random_pairs(
        self,
        rng: random.Random,
        num_negatives: int,
        append: bool = False,
    ) -> None:
        """
        Create pairs with randomly sampled incorrect items.

        Args:
            rng: Random number generator.
            num_negatives: Number of random negatives per positive.
            append: Whether to append to existing pairs or replace them.
        """
        if not append:
            self.preference_pairs = []

        for i in range(len(self.prompts)):
            chosen = self.chosen_items[i]

            for _ in range(num_negatives):
                # Sample a random item that isn't the correct answer
                rejected = chosen
                attempts = 0
                while not self._items_differ(chosen, rejected) and attempts < 50:
                    rejected = rng.choice(self.all_items)
                    attempts += 1

                if self._items_differ(chosen, rejected):
                    self.preference_pairs.append({
                        "prompt": self._format_dpo_prompt(self.prompts[i]),
                        "chosen": self._format_response(chosen),
                        "rejected": self._format_response(rejected),
                    })

    def generate_hard_negatives(
        self,
        similarity_threshold: float = 0.3,
        random_state: int = 42,
    ) -> List[Dict[str, str]]:
        """
        Generate hard negative pairs using item similarity.

        Hard negatives are items that share words with the correct answer
        but are different products. These provide stronger training signal
        than random negatives.

        Args:
            similarity_threshold: Minimum word overlap ratio for hard negatives.
            random_state: Random seed.

        Returns:
            List of hard negative preference pairs.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        rng = random.Random(random_state)
        hard_pairs = []

        for i in range(len(self.prompts)):
            chosen = self.chosen_items[i]
            chosen_words = set(chosen.lower().split())

            # Find items with partial word overlap (hard negatives)
            candidates = []
            for item in self.all_items:
                if self._items_differ(chosen, item):
                    item_words = set(item.lower().split())
                    if chosen_words and item_words:
                        overlap = len(chosen_words & item_words) / max(
                            len(chosen_words), 1
                        )
                        if overlap >= similarity_threshold:
                            candidates.append(item)

            if candidates:
                rejected = rng.choice(candidates)
                hard_pairs.append({
                    "prompt": self._format_dpo_prompt(self.prompts[i]),
                    "chosen": self._format_response(chosen),
                    "rejected": self._format_response(rejected),
                })

        logger.info(f"Generated {len(hard_pairs)} hard negative pairs")
        return hard_pairs

    @staticmethod
    def _items_differ(item_a: str, item_b: str) -> bool:
        """Check if two items are meaningfully different."""
        a_clean = item_a.strip().lower()
        b_clean = item_b.strip().lower()
        return a_clean != b_clean

    @staticmethod
    def _format_dpo_prompt(prompt: str) -> str:
        """
        Format a prompt for DPO training.

        Wraps the original prompt in an instruction template.

        Args:
            prompt: Original recommendation prompt.

        Returns:
            Formatted prompt string.
        """
        return (
            f"### Instruction:\n"
            f"You are a product recommendation assistant. Based on the "
            f"user's purchase history, recommend the next item they should buy.\n\n"
            f"### Input:\n{prompt}\n\n"
            f"### Response:\n"
        )

    @staticmethod
    def _format_response(item: str) -> str:
        """
        Format a response item for DPO training.

        Args:
            item: Product name/title.

        Returns:
            Formatted response string.
        """
        return f"Based on the user's purchase history, I recommend: {item}"

    def to_hf_dataset(self) -> Any:
        """
        Convert preference pairs to a HuggingFace Dataset.

        Returns:
            HuggingFace Dataset with columns: prompt, chosen, rejected.

        Raises:
            RuntimeError: If no pairs have been generated.
            ImportError: If datasets library isn't installed.
        """
        if not self.preference_pairs:
            raise RuntimeError(
                "No preference pairs generated. Call generate_pairs() first."
            )

        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets required. "
                "Install with: pip install datasets"
            )

        dataset = Dataset.from_list(self.preference_pairs)
        logger.info(f"Created HuggingFace Dataset with {len(dataset)} samples")
        return dataset

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert preference pairs to a pandas DataFrame.

        Returns:
            DataFrame with columns: prompt, chosen, rejected.
        """
        if not self.preference_pairs:
            raise RuntimeError(
                "No preference pairs generated. Call generate_pairs() first."
            )
        return pd.DataFrame(self.preference_pairs)

    def split_pairs(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split preference pairs into train/val/test sets.

        Args:
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            random_state: Random seed.

        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs).
        """
        if not self.preference_pairs:
            raise RuntimeError("No pairs generated. Call generate_pairs() first.")

        rng = random.Random(random_state)
        pairs = self.preference_pairs.copy()
        rng.shuffle(pairs)

        n = len(pairs)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        train = pairs[:n_train]
        val = pairs[n_train : n_train + n_val]
        test = pairs[n_train + n_val :]

        logger.info(
            f"Preference split — Train: {len(train)} | "
            f"Val: {len(val)} | Test: {len(test)}"
        )
        return train, val, test

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the generated preference data.

        Returns:
            Dictionary with data statistics.
        """
        stats: Dict[str, Any] = {
            "num_prompts": len(self.prompts),
            "num_ground_truth": len(self.chosen_items),
            "num_predictions": len(self.model_predictions),
            "num_unique_items": len(self.all_items),
            "num_preference_pairs": len(self.preference_pairs),
        }

        if self.preference_pairs:
            # Compute average prompt/response lengths
            prompt_lens = [len(p["prompt"].split()) for p in self.preference_pairs]
            chosen_lens = [len(p["chosen"].split()) for p in self.preference_pairs]
            rejected_lens = [len(p["rejected"].split()) for p in self.preference_pairs]

            stats.update({
                "avg_prompt_length_words": float(np.mean(prompt_lens)),
                "avg_chosen_length_words": float(np.mean(chosen_lens)),
                "avg_rejected_length_words": float(np.mean(rejected_lens)),
            })

        if self._is_loaded:
            # Model accuracy (how often prediction matches ground truth)
            correct = sum(
                1 for c, p in zip(self.chosen_items, self.model_predictions)
                if not self._items_differ(c, p)
            )
            stats["model_accuracy"] = correct / max(len(self.chosen_items), 1)
            stats["model_errors"] = len(self.chosen_items) - correct

        return stats

    def __repr__(self) -> str:
        if self._is_loaded:
            return (
                f"PreferenceDataGenerator("
                f"prompts={len(self.prompts)}, "
                f"pairs={len(self.preference_pairs)})"
            )
        return f"PreferenceDataGenerator(data_dir='{self.data_dir}', loaded=False)"
