"""
Unit tests for DPO training components.

Tests cover:
- DPO config validation and serialization
- LoRA config conversion
- DPOTrainer initialization (without loading actual model)
"""

import json
import os
import tempfile

import pytest

from src.train.dpo_config import DPOTrainingConfig, LoRAConfig


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_default_values(self):
        """Test that defaults are sensible."""
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules

    def test_to_peft_config(self):
        """Test conversion to PEFT-compatible dict."""
        config = LoRAConfig(rank=8, alpha=16)
        peft_dict = config.to_peft_config()

        assert peft_dict["r"] == 8
        assert peft_dict["lora_alpha"] == 16
        assert peft_dict["task_type"] == "CAUSAL_LM"

    def test_custom_target_modules(self):
        """Test custom target module specification."""
        config = LoRAConfig(target_modules=["q_proj", "v_proj"])
        assert len(config.target_modules) == 2


class TestDPOTrainingConfig:
    """Tests for DPO training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DPOTrainingConfig()
        assert config.beta == 0.1
        assert config.num_epochs == 3
        assert config.learning_rate == 5e-5
        assert config.loss_type == "sigmoid"

    def test_effective_batch_size(self):
        """Test effective batch size computation."""
        config = DPOTrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=8,
        )
        assert config.effective_batch_size == 32

    def test_validate_success(self):
        """Test that valid config passes validation."""
        config = DPOTrainingConfig()
        config.validate()  # Should not raise

    def test_validate_negative_beta_raises(self):
        """Test that negative beta fails validation."""
        config = DPOTrainingConfig(beta=-0.1)
        with pytest.raises(ValueError, match="beta"):
            config.validate()

    def test_validate_prompt_length_exceeds_max_raises(self):
        """Test that prompt length > max length fails."""
        config = DPOTrainingConfig(
            max_prompt_length=512,
            max_length=256,
        )
        with pytest.raises(ValueError, match="max_prompt_length"):
            config.validate()

    def test_validate_both_quantizations_raises(self):
        """Test that using both 4-bit and 8-bit fails."""
        config = DPOTrainingConfig(use_4bit=True, use_8bit=True)
        with pytest.raises(ValueError, match="4-bit"):
            config.validate()

    def test_validate_both_precision_raises(self):
        """Test that using both bf16 and fp16 fails."""
        config = DPOTrainingConfig(bf16=True, fp16=True)
        with pytest.raises(ValueError, match="bf16"):
            config.validate()

    def test_save_and_load(self, tmp_path):
        """Test config serialization round-trip."""
        config = DPOTrainingConfig(
            beta=0.2,
            learning_rate=1e-4,
            num_epochs=5,
        )

        save_path = str(tmp_path / "config.json")
        config.save(save_path)

        loaded = DPOTrainingConfig.load(save_path)

        assert loaded.beta == 0.2
        assert loaded.learning_rate == 1e-4
        assert loaded.num_epochs == 5
        assert loaded.lora.rank == 16  # Default preserved

    def test_to_training_args(self):
        """Test conversion to TrainingArguments dict."""
        config = DPOTrainingConfig(
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=8,
        )
        args_dict = config.to_training_args_dict()

        assert args_dict["learning_rate"] == 1e-4
        assert args_dict["num_train_epochs"] == 5
        assert args_dict["per_device_train_batch_size"] == 8   
        assert args_dict["remove_unused_columns"] is False

    def test_repr(self):
        """Test string representation."""
        config = DPOTrainingConfig()
        repr_str = repr(config)
        assert "DPOTrainingConfig" in repr_str
        assert "beta=0.1" in repr_str
