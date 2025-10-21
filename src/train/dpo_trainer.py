"""
DPO Trainer for Recommendation LLM Alignment.

Wraps HuggingFace TRL's DPOTrainer with recommendation-specific
enhancements: custom data collation, reward tracking, and LoRA
adapter management.

Direct Preference Optimization trains the model to prefer ground-truth
recommendations over incorrect ones without needing a separate reward model.

Loss: L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

Where:
    - π: Policy (model being trained)
    - π_ref: Reference model (frozen copy)
    - y_w: Chosen (preferred) response
    - y_l: Rejected response
    - β: Temperature controlling preference sharpness

References:
    - Rafailov et al., "Direct Preference Optimization" (2023)
    - TRL Library: https://github.com/huggingface/trl
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from src.train.dpo_config import DPOTrainingConfig, LoRAConfig


class RecommendationDPOTrainer:
    """
    DPO trainer specialized for recommendation LLM alignment.

    Manages the full DPO training lifecycle:
    1. Load and quantize base model
    2. Apply LoRA adapters
    3. Configure reference model
    4. Run DPO training with preference data
    5. Track rewards and evaluate alignment

    Attributes:
        config: DPO training configuration.
        model: Base language model with LoRA adapters.
        ref_model: Frozen reference model for KL penalty.
        tokenizer: Model tokenizer.
        is_initialized: Whether model and tokenizer are loaded.

    Example:
        >>> config = DPOTrainingConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> trainer = RecommendationDPOTrainer(config)
        >>> trainer.initialize()
        >>> trainer.train(train_dataset, eval_dataset)
        >>> metrics = trainer.get_training_metrics()
    """

    def __init__(self, config: DPOTrainingConfig) -> None:
        """
        Initialize DPO trainer with configuration.

        Args:
            config: DPO training configuration.
        """
        config.validate()

        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        self.is_initialized = False
        self._training_metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "rewards_chosen": [],
            "rewards_rejected": [],
            "reward_margins": [],
            "reward_accuracies": [],
        }

        logger.info(f"RecommendationDPOTrainer created | Config: {config}")

    def initialize(self) -> None:
        """
        Load model, tokenizer, and apply LoRA adapters.

        This is separated from __init__ for lazy loading — heavy
        model loading only happens when explicitly requested.

        Raises:
            ImportError: If required libraries aren't installed.
            RuntimeError: If model loading fails.
        """
        if self.is_initialized:
            logger.info("Already initialized, skipping.")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError(
                "Required libraries not installed. Run: "
                "pip install transformers peft trl bitsandbytes"
            ) from e

        logger.info(f"Loading model: {self.config.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Required for DPO

        # Load model with quantization
        model_kwargs = self._get_model_kwargs()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Prepare for quantized training if needed
        if self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA adapters
        lora_config = LoraConfig(**self.config.lora.to_peft_config())
        self.model = get_peft_model(self.model, lora_config)

        # Log trainable parameters
        trainable, total = self._count_parameters()
        logger.info(
            f"Model loaded | "
            f"Total params: {total:,} | "
            f"Trainable params: {trainable:,} | "
            f"Trainable %: {100 * trainable / total:.2f}%"
        )

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.is_initialized = True

    def _get_model_kwargs(self) -> Dict:
        """Build model loading kwargs based on config."""
        kwargs: Dict[str, Any] = {"trust_remote_code": True}

        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                logger.warning("BitsAndBytes not available. Loading in fp16.")
                kwargs["torch_dtype"] = torch.float16
        elif self.config.use_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16

        return kwargs

    def _count_parameters(self) -> Tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run DPO training on preference dataset.

        Args:
            train_dataset: HuggingFace Dataset with columns:
                prompt, chosen, rejected.
            eval_dataset: Optional validation dataset.

        Returns:
            Training results dictionary with metrics.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call initialize() first.")

        try:
            from trl import DPOTrainer, DPOConfig
        except ImportError:
            raise ImportError("TRL required. Install with: pip install trl")

        logger.info(
            f"Starting DPO training | "
            f"Train: {len(train_dataset)} samples | "
            f"Eval: {len(eval_dataset) if eval_dataset else 0} samples | "
            f"Beta: {self.config.beta}"
        )

        # Configure DPO training arguments
        training_args = DPOConfig(
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            **self.config.to_training_args_dict(),
        )

        # Create TRL DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,  # None = implicit reference model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        # Train
        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time

        # Save final model
        self._save_model()

        results = {
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
            "epochs_completed": self.config.num_epochs,
        }

        logger.info(
            f"DPO training complete | "
            f"Loss: {train_result.training_loss:.4f} | "
            f"Time: {training_time / 60:.1f} minutes"
        )

        return results

    def _save_model(self) -> None:
        """Save the trained LoRA adapter and tokenizer."""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA adapter (not full model)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        self.config.save(os.path.join(output_dir, "dpo_config.json"))

        logger.info(f"Model saved to {output_dir}")

    @torch.no_grad()
    def compute_rewards(
        self,
        dataset: Any,
        batch_size: int = 8,
    ) -> Dict[str, np.ndarray]:
        """
        Compute reward scores for a preference dataset.

        For each (chosen, rejected) pair, computes the implicit reward
        as the difference in log-probabilities.

        Args:
            dataset: HuggingFace Dataset with preference pairs.
            batch_size: Processing batch size.

        Returns:
            Dictionary with reward arrays:
            - chosen_rewards: Rewards for chosen responses
            - rejected_rewards: Rewards for rejected responses
            - margins: Chosen - Rejected reward differences
        """
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call initialize() first.")

        self.model.eval()
        chosen_rewards = []
        rejected_rewards = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            for j in range(len(batch["prompt"])):
                prompt = batch["prompt"][j]
                chosen = batch["chosen"][j]
                rejected = batch["rejected"][j]

                chosen_score = self._compute_log_prob(prompt, chosen)
                rejected_score = self._compute_log_prob(prompt, rejected)

                chosen_rewards.append(chosen_score)
                rejected_rewards.append(rejected_score)

        chosen_arr = np.array(chosen_rewards)
        rejected_arr = np.array(rejected_rewards)
        margins = chosen_arr - rejected_arr

        reward_accuracy = float(np.mean(margins > 0))

        logger.info(
            f"Reward computation | "
            f"Mean margin: {margins.mean():.4f} | "
            f"Reward accuracy: {reward_accuracy:.4f}"
        )

        return {
            "chosen_rewards": chosen_arr,
            "rejected_rewards": rejected_arr,
            "margins": margins,
            "reward_accuracy": reward_accuracy,
        }

    def _compute_log_prob(self, prompt: str, response: str) -> float:
        """Compute log probability of response given prompt."""
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

        # Sum log probs of response tokens only
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_tokens["input_ids"].shape[1]

        response_log_probs = []
        for pos in range(prompt_len, inputs["input_ids"].shape[1] - 1):
            next_token = inputs["input_ids"][0, pos + 1]
            response_log_probs.append(log_probs[0, pos, next_token].item())

        return float(np.mean(response_log_probs)) if response_log_probs else 0.0

    def generate_recommendation(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a recommendation using the DPO-aligned model.

        Args:
            prompt: Recommendation prompt.
            max_new_tokens: Maximum response length.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated recommendation text.
        """
        if not self.is_initialized:
            raise RuntimeError("Not initialized. Call initialize() first.")

        self.model.eval()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_length,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return response.strip()

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        if self.trainer is None:
            return {"status": "not_trained"}

        state = self.trainer.state
        return {
            "epochs_completed": state.epoch,
            "global_steps": state.global_step,
            "best_metric": state.best_metric,
            "log_history": state.log_history[-10:] if state.log_history else [],
        }

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "not_initialized"
        return (
            f"RecommendationDPOTrainer("
            f"model='{self.config.model_name}', "
            f"beta={self.config.beta}, "
            f"status={status})"
        )
