"""
Amazon Review Dataset Loader for GNN-LLM Recommendation System.

Handles loading, preprocessing, and splitting of the Amazon Prime Pantry
review dataset for use in graph construction and model training.

Data Sources:
    - final_dataset.xlsx: Full interaction dataset with user IDs and product ASINs
    - ground_truth.csv: Ground truth labels for evaluation
    - final_generated_output.csv: Model-generated predictions for comparison
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class AmazonDatasetLoader:
    """
    Load and preprocess Amazon Prime Pantry review data for recommendation.

    This loader handles the full pipeline from raw Excel/CSV data to
    structured DataFrames suitable for graph construction and model training.

    Attributes:
        data_dir: Root directory containing data files.
        interactions_df: Processed interactions DataFrame.
        user_id_map: Mapping from string user IDs to integer indices.
        item_id_map: Mapping from string item ASINs to integer indices.

    Example:
        >>> loader = AmazonDatasetLoader("./")
        >>> loader.load_data()
        >>> train_df, val_df, test_df = loader.split_data()
        >>> print(f"Users: {loader.num_users}, Items: {loader.num_items}")
    """

    def __init__(self, data_dir: str) -> None:
        """
        Initialize the dataset loader.

        Args:
            data_dir: Path to directory containing the data files.

        Raises:
            FileNotFoundError: If the data directory does not exist.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.data_dir = data_dir
        self.interactions_df: Optional[pd.DataFrame] = None
        self.items_df: Optional[pd.DataFrame] = None
        self.user_id_map: Dict[str, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self._is_loaded = False

        logger.info(f"Initialized AmazonDatasetLoader with data_dir={data_dir}")

    @property
    def num_users(self) -> int:
        """Return the number of unique users in the dataset."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return len(self.user_id_map)

    @property
    def num_items(self) -> int:
        """Return the number of unique items in the dataset."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return len(self.item_id_map)

    @property
    def num_interactions(self) -> int:
        """Return the total number of user-item interactions."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return len(self.interactions_df)

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the Amazon review dataset.

        Reads the Excel dataset, extracts user-item interactions,
        builds ID mappings, and creates the core interactions DataFrame.

        Returns:
            DataFrame with columns: [user_id, item_id, user_idx, item_idx, title]

        Raises:
            FileNotFoundError: If required data files are missing.
            ValueError: If data files are empty or malformed.
        """
        xlsx_path = os.path.join(self.data_dir, "final_dataset.xlsx")
        if not os.path.isfile(xlsx_path):
            raise FileNotFoundError(
                f"Dataset file not found: {xlsx_path}. "
                "Ensure 'final_dataset.xlsx' is in the data directory."
            )

        logger.info(f"Loading dataset from {xlsx_path}...")
        raw_df = pd.read_excel(xlsx_path, engine="openpyxl")

        if raw_df.empty:
            raise ValueError("Dataset file is empty.")

        logger.info(f"Raw dataset shape: {raw_df.shape}")
        logger.info(f"Columns: {list(raw_df.columns)}")

        # Process the dataset based on its structure
        interactions = self._extract_interactions(raw_df)

        if interactions.empty:
            raise ValueError("No valid interactions extracted from dataset.")

        # Build ID mappings (string IDs → contiguous integers)
        self.user_id_map = self._build_id_mapping(
            interactions["user_id"].unique()
        )
        self.item_id_map = self._build_id_mapping(
            interactions["item_id"].unique()
        )

        # Add integer index columns
        interactions["user_idx"] = interactions["user_id"].map(self.user_id_map)
        interactions["item_idx"] = interactions["item_id"].map(self.item_id_map)

        self.interactions_df = interactions.reset_index(drop=True)
        self._is_loaded = True

        logger.info(
            f"Loaded {self.num_interactions} interactions | "
            f"{self.num_users} users | {self.num_items} items"
        )

        return self.interactions_df

    def _extract_interactions(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract user-item interaction pairs from the raw dataset.

        The dataset contains grouped user purchase histories. This method
        parses each user's history to create individual interaction records.

        Args:
            raw_df: Raw DataFrame loaded from Excel.

        Returns:
            DataFrame with columns: [user_id, item_id, title, interaction_order]
        """
        interactions: List[Dict] = []

        # Check if the DataFrame has the expected structure
        # The Excel file may have columns like: ID, info, item_1_title, etc.
        if "ID" in raw_df.columns:
            logger.info("Detected grouped format with user IDs.")
            return self._parse_grouped_format(raw_df)

        # Alternative: direct reviewer-item format
        if "reviewerID" in raw_df.columns and "asin" in raw_df.columns:
            logger.info("Detected direct reviewer-item format.")
            df = raw_df[["reviewerID", "asin"]].copy()
            df.columns = ["user_id", "item_id"]

            if "title" in raw_df.columns:
                df["title"] = raw_df["title"].values
            else:
                df["title"] = df["item_id"]

            df["interaction_order"] = df.groupby("user_id").cumcount()
            return df

        # Fallback: try to use prompts/outputs columns
        if "prompts" in raw_df.columns:
            logger.info("Detected prompts format. Creating synthetic interactions.")
            return self._parse_prompts_format(raw_df)

        raise ValueError(
            f"Unrecognized dataset format. Columns: {list(raw_df.columns)}"
        )

    def _parse_grouped_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the grouped format where each row is a user with nested item data.

        Args:
            raw_df: DataFrame with 'ID' column and item title columns.

        Returns:
            Flattened interactions DataFrame.
        """
        interactions: List[Dict] = []

        # Identify item title columns (item_1_title, item_2_title, etc.)
        title_cols = [c for c in raw_df.columns if c.startswith("item_") and c.endswith("_title")]

        if title_cols:
            for _, row in raw_df.iterrows():
                user_id = str(row["ID"])
                for order, col in enumerate(sorted(title_cols)):
                    title = str(row[col]) if pd.notna(row[col]) else ""
                    if title and title != "nan":
                        # Use a sanitized title as a proxy item_id
                        item_id = f"item_{hash(title) % 100000:05d}"
                        interactions.append({
                            "user_id": user_id,
                            "item_id": item_id,
                            "title": title,
                            "interaction_order": order,
                        })
        else:
            # Try to parse the 'info' column if it exists
            if "info" in raw_df.columns:
                for _, row in raw_df.iterrows():
                    user_id = str(row["ID"])
                    info = row["info"]
                    if isinstance(info, dict) and "data" in info:
                        data = info["data"]
                        total = data.get("total_asin", 0)
                        for i in range(1, total + 1):
                            item_data = data.get(str(i), {})
                            asin = item_data.get("asin", f"unknown_{i}")
                            title = item_data.get("title", asin)
                            interactions.append({
                                "user_id": user_id,
                                "item_id": asin,
                                "title": title,
                                "interaction_order": i - 1,
                            })

        df = pd.DataFrame(interactions)
        if df.empty:
            logger.warning("No interactions extracted from grouped format.")
        return df

    def _parse_prompts_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse prompt-based format to create synthetic user-item interactions.

        Each row's prompt contains a user's purchase history as text.
        We create synthetic user IDs and extract item titles.

        Args:
            raw_df: DataFrame with 'prompts' and 'outputs' columns.

        Returns:
            Interactions DataFrame derived from prompts.
        """
        interactions: List[Dict] = []

        for idx, row in raw_df.iterrows():
            user_id = f"user_{idx:06d}"
            prompt = str(row.get("prompts", ""))
            output = str(row.get("outputs", ""))

            # Extract items from "Reviewer has bought X, Y, Z" pattern
            if "has bought" in prompt:
                # Split on commas and extract item names
                bought_section = prompt.split("has bought")[-1]
                consideration_split = bought_section.split("Considering")
                items_text = consideration_split[0] if consideration_split else bought_section

                items = [
                    item.strip().strip(",").strip()
                    for item in items_text.split(",")
                    if item.strip() and len(item.strip()) > 2
                ]

                for order, item_title in enumerate(items):
                    item_id = f"item_{hash(item_title) % 100000:05d}"
                    interactions.append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "title": item_title,
                        "interaction_order": order,
                    })

            # Add the ground truth output as the last interaction
            if output and output != "nan":
                item_id = f"item_{hash(output) % 100000:05d}"
                interactions.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "title": output,
                    "interaction_order": len(interactions),
                })

        return pd.DataFrame(interactions)

    @staticmethod
    def _build_id_mapping(ids: np.ndarray) -> Dict[str, int]:
        """
        Create a mapping from string IDs to contiguous integer indices.

        Args:
            ids: Array of unique string identifiers.

        Returns:
            Dictionary mapping each string ID to a unique integer index.
        """
        sorted_ids = sorted(ids)
        return {str(id_): idx for idx, id_ in enumerate(sorted_ids)}

    def split_data(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into train/validation/test sets.

        Uses temporal splitting where each user's last interaction goes to
        test, second-to-last to validation, and the rest to training.
        Falls back to random splitting if users have too few interactions.

        Args:
            val_ratio: Fraction of data for validation (used in fallback).
            test_ratio: Fraction of data for testing (used in fallback).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (train_df, val_df, test_df).

        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        df = self.interactions_df.copy()

        # Temporal split: for each user, last item = test, second-to-last = val
        train_records = []
        val_records = []
        test_records = []

        for user_id, group in df.groupby("user_id"):
            sorted_group = group.sort_values("interaction_order")

            if len(sorted_group) >= 3:
                train_records.append(sorted_group.iloc[:-2])
                val_records.append(sorted_group.iloc[-2:-1])
                test_records.append(sorted_group.iloc[-1:])
            elif len(sorted_group) == 2:
                train_records.append(sorted_group.iloc[:1])
                test_records.append(sorted_group.iloc[1:])
            else:
                train_records.append(sorted_group)

        train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
        val_df = pd.concat(val_records, ignore_index=True) if val_records else pd.DataFrame()
        test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()

        logger.info(
            f"Data split — Train: {len(train_df)} | "
            f"Val: {len(val_df)} | Test: {len(test_df)}"
        )

        return train_df, val_df, test_df

    def get_user_item_matrix(self) -> np.ndarray:
        """
        Create a sparse user-item interaction matrix.

        Returns:
            Binary matrix of shape (num_users, num_items) where entry
            (i, j) = 1 indicates user i interacted with item j.

        Raises:
            RuntimeError: If data hasn't been loaded yet.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)

        for _, row in self.interactions_df.iterrows():
            u_idx = row["user_idx"]
            i_idx = row["item_idx"]
            matrix[u_idx, i_idx] = 1.0

        density = matrix.sum() / (self.num_users * self.num_items)
        logger.info(f"User-item matrix density: {density:.6f}")

        return matrix

    def get_summary(self) -> Dict:
        """
        Get a summary of the loaded dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        interactions_per_user = self.interactions_df.groupby("user_id").size()
        interactions_per_item = self.interactions_df.groupby("item_id").size()

        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_interactions": self.num_interactions,
            "avg_interactions_per_user": float(interactions_per_user.mean()),
            "median_interactions_per_user": float(interactions_per_user.median()),
            "max_interactions_per_user": int(interactions_per_user.max()),
            "min_interactions_per_user": int(interactions_per_user.min()),
            "avg_interactions_per_item": float(interactions_per_item.mean()),
            "density": self.num_interactions / (self.num_users * self.num_items),
        }

    def __repr__(self) -> str:
        if self._is_loaded:
            return (
                f"AmazonDatasetLoader("
                f"users={self.num_users}, "
                f"items={self.num_items}, "
                f"interactions={self.num_interactions})"
            )
        return f"AmazonDatasetLoader(data_dir='{self.data_dir}', loaded=False)"
