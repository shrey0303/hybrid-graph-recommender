"""
Unit tests for PreferenceDataGenerator.

Tests cover:
- Data loading from CSV/Excel files
- Preference pair generation (model-based, random, mixed)
- Hard negative mining
- Data validation (chosen ≠ rejected)
- Split functionality
- Statistics computation
"""

import os
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.data.preference_data_generator import PreferenceDataGenerator


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data files."""
    # Create ground_truth.csv
    ground_truth = [
        "Organic Peanut Butter",
        "Whole Wheat Pasta",
        "Green Tea Bags",
        "Almond Milk",
        "Olive Oil Extra Virgin",
        "Brown Rice",
        "Honey Raw Organic",
        "Dark Chocolate Bar",
        "Coconut Water",
        "Oatmeal Instant",
    ]
    gt_df = pd.DataFrame(ground_truth)
    gt_df.to_csv(tmp_path / "ground_truth.csv", index=False, header=False)

    # Create final_generated_output.csv (model predictions - some wrong)
    predictions = [
        "Organic Peanut Butter",       # Correct
        "White Pasta Regular",          # Wrong
        "Green Tea Bags",               # Correct
        "Soy Milk Regular",             # Wrong
        "Canola Oil",                   # Wrong
        "White Rice Jasmine",           # Wrong
        "Honey Raw Organic",            # Correct
        "Milk Chocolate Bar",           # Wrong
        "Orange Juice",                 # Wrong
        "Oatmeal Instant",              # Correct
    ]
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(tmp_path / "final_generated_output.csv", index=False, header=False)

    # Create final_dataset.xlsx with prompts
    prompts = [
        f"User has bought item_{i}. What should they buy next?"
        for i in range(10)
    ]
    xlsx_df = pd.DataFrame({"prompts": prompts})
    xlsx_df.to_excel(tmp_path / "final_dataset.xlsx", index=False)

    return str(tmp_path)


@pytest.fixture
def generator(sample_data_dir):
    """Create a loaded PreferenceDataGenerator."""
    gen = PreferenceDataGenerator(sample_data_dir)
    gen.load_data()
    return gen


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_data_success(self, sample_data_dir):
        """Test that data loads successfully from temp directory."""
        gen = PreferenceDataGenerator(sample_data_dir)
        gen.load_data()

        assert gen._is_loaded
        assert len(gen.prompts) == 10
        assert len(gen.chosen_items) == 10
        assert len(gen.model_predictions) == 10

    def test_load_data_creates_catalog(self, generator):
        """Test that item catalog is built from all items."""
        assert len(generator.all_items) > 0
        assert "Organic Peanut Butter" in generator.all_items

    def test_missing_directory_raises(self):
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PreferenceDataGenerator("/nonexistent/path")

    def test_missing_ground_truth_raises(self, tmp_path):
        """Test that missing ground_truth.csv raises FileNotFoundError."""
        # Create only predictions, no ground truth
        pd.DataFrame(["item"]).to_csv(
            tmp_path / "final_generated_output.csv", index=False, header=False
        )
        gen = PreferenceDataGenerator(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Ground truth"):
            gen.load_data()


class TestPairGeneration:
    """Tests for preference pair generation."""

    def test_model_based_pairs(self, generator):
        """Test model-based pair generation creates valid pairs."""
        pairs = generator.generate_pairs(strategy="model_based")

        assert len(pairs) > 0
        # Should only have pairs where model was wrong (6 out of 10)
        assert len(pairs) == 6

    def test_random_pairs(self, generator):
        """Test random pair generation."""
        pairs = generator.generate_pairs(strategy="random")
        assert len(pairs) > 0

    def test_mixed_pairs(self, generator):
        """Test mixed strategy generates more pairs than model-based alone."""
        model_pairs = generator.generate_pairs(strategy="model_based")
        model_count = len(model_pairs)

        mixed_pairs = generator.generate_pairs(strategy="mixed")
        # Mixed should have model-based + random
        assert len(mixed_pairs) >= model_count

    def test_chosen_differs_from_rejected(self, generator):
        """Test that chosen ≠ rejected for all pairs."""
        pairs = generator.generate_pairs(strategy="model_based")

        for pair in pairs:
            chosen_clean = pair["chosen"].lower().strip()
            rejected_clean = pair["rejected"].lower().strip()
            assert chosen_clean != rejected_clean, (
                f"Chosen and rejected are identical: {pair['chosen']}"
            )

    def test_pairs_have_required_keys(self, generator):
        """Test that each pair has prompt, chosen, rejected keys."""
        pairs = generator.generate_pairs(strategy="model_based")

        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair
            assert len(pair["prompt"]) > 0
            assert len(pair["chosen"]) > 0
            assert len(pair["rejected"]) > 0

    def test_invalid_strategy_raises(self, generator):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            generator.generate_pairs(strategy="invalid")

    def test_generate_before_load_raises(self, sample_data_dir):
        """Test that generating pairs before loading raises error."""
        gen = PreferenceDataGenerator(sample_data_dir)
        with pytest.raises(RuntimeError, match="not loaded"):
            gen.generate_pairs()


class TestHardNegatives:
    """Tests for hard negative mining."""

    def test_hard_negatives_generation(self, generator):
        """Test hard negative pair generation."""
        pairs = generator.generate_hard_negatives(similarity_threshold=0.1)
        # May or may not find hard negatives depending on item overlap
        assert isinstance(pairs, list)

    def test_hard_negatives_validity(self, generator):
        """Test that hard negatives have proper keys."""
        pairs = generator.generate_hard_negatives(similarity_threshold=0.1)
        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair


class TestDataSplitting:
    """Tests for train/val/test splitting."""

    def test_split_produces_three_sets(self, generator):
        """Test that split produces train, val, test."""
        generator.generate_pairs(strategy="model_based")
        train, val, test = generator.split_pairs()

        assert len(train) > 0
        assert len(val) > 0 or len(test) > 0  # At least one non-empty
        assert len(train) + len(val) + len(test) == len(generator.preference_pairs)

    def test_split_before_generate_raises(self, generator):
        """Test that splitting before generating raises error."""
        with pytest.raises(RuntimeError, match="No pairs"):
            generator.split_pairs()


class TestStatistics:
    """Tests for statistics computation."""

    def test_statistics_keys(self, generator):
        """Test that statistics have expected keys."""
        generator.generate_pairs(strategy="model_based")
        stats = generator.get_statistics()

        assert "num_prompts" in stats
        assert "num_preference_pairs" in stats
        assert "model_accuracy" in stats
        assert "num_unique_items" in stats

    def test_model_accuracy(self, generator):
        """Test model accuracy computation."""
        stats = generator.get_statistics()
        # 4 correct out of 10 = 0.4
        assert 0.3 <= stats["model_accuracy"] <= 0.5

    def test_to_dataframe(self, generator):
        """Test conversion to DataFrame."""
        generator.generate_pairs(strategy="model_based")
        df = generator.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "prompt" in df.columns
        assert "chosen" in df.columns
        assert "rejected" in df.columns
        assert len(df) == len(generator.preference_pairs)

    def test_repr(self, generator):
        """Test string representation."""
        repr_str = repr(generator)
        assert "PreferenceDataGenerator" in repr_str
