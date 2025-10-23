"""
DPO Training Configuration.

Dataclass-based configuration for Direct Preference Optimization training,
including model, LoRA adapter, training hyperparameters, and logging settings.

Provides sensible defaults tuned for the recommendation domain with
TinyLlama/Mistral-7B base models.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json

from loguru import logger


@dataclass
class LoRAConfig:
    """
    Configuration for Low-Rank Adaptation (LoRA) adapters.

    Controls which model layers are adapted and with what capacity.
    Lower rank = fewer trainable parameters = faster training.

    Attributes:
        rank: LoRA rank (bottleneck dimension). Higher = more capacity.
        alpha: LoRA scaling factor. Typically 2x rank.
        dropout: Dropout on LoRA layers.
        target_modules: Model layers to apply LoRA to.
        bias: Whether to train bias parameters ('none', 'all', 'lora_only').
    """
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> Dict:
        """Convert to PEFT LoraConfig-compatible dict."""
        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class DPOTrainingConfig:
    """
    Full configuration for DPO training pipeline.

    Groups all hyperparameters into logical sections:
    model, LoRA adapter, training loop, and DPO-specific settings.

    Attributes:
        model_name: HuggingFace model identifier.
        lora: LoRA adapter configuration.
        beta: DPO temperature (controls preference sharpness).
        learning_rate: Optimizer learning rate.
        num_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Gradient accumulation for effective batch size.
        max_length: Maximum token sequence length.
        max_prompt_length: Maximum prompt length (truncation).
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: L2 regularization.
        logging_steps: Steps between log messages.
        save_steps: Steps between checkpoint saves.
        output_dir: Directory for checkpoints and logs.
        use_4bit: Whether to use 4-bit quantization (QLoRA).
        use_8bit: Whether to use 8-bit quantization.
        bf16: Use bfloat16 mixed precision.
        fp16: Use float16 mixed precision.
        gradient_checkpointing: Reduce memory at cost of speed.
        seed: Random seed for reproducibility.

    Example:
        >>> config = DPOTrainingConfig(
        ...     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     beta=0.1,
        ...     num_epochs=3,
        ... )
        >>> config.save("config.json")
    """

    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    reference_model_name: Optional[str] = None  # None = use same as model_name

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # DPO-specific
    beta: float = 0.1  # Temperature parameter (lower = sharper preferences)
    loss_type: str = "sigmoid"  # DPO loss variant: 'sigmoid', 'hinge', 'ipo'
    label_smoothing: float = 0.0  # Label smoothing for robustness

    # Training
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    max_prompt_length: int = 384
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Precision
    use_4bit: bool = False
    use_8bit: bool = True
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Logging & Checkpoints
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "models/dpo_checkpoints"
    logging_dir: str = "logs/dpo"

    # Reproducibility
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        """
        Validate configuration for common issues.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_prompt_length >= self.max_length:
            raise ValueError(
                f"max_prompt_length ({self.max_prompt_length}) must be < "
                f"max_length ({self.max_length})"
            )
        if self.use_4bit and self.use_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16")

        logger.info(f"Config validated | Effective batch size: {self.effective_batch_size}")

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DPOTrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Handle nested LoRA config
        lora_data = data.pop("lora", {})
        config = cls(**data)
        config.lora = LoRAConfig(**lora_data)
        return config

    def to_training_args_dict(self) -> Dict:
        """Convert to TrainingArguments-compatible dict."""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "logging_dir": self.logging_dir,
            "remove_unused_columns": False,
        }

    def __repr__(self) -> str:
        return (
            f"DPOTrainingConfig("
            f"model='{self.model_name}', "
            f"beta={self.beta}, "
            f"lr={self.learning_rate}, "
            f"epochs={self.num_epochs}, "
            f"lora_rank={self.lora.rank}, "
            f"eff_batch={self.effective_batch_size})"
        )
