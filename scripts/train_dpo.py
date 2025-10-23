#!/usr/bin/env python3
"""
DPO Training Entry Point.

Generates preference pairs from Amazon dataset and trains the 
recommendation LLM using Direct Preference Optimization.

Usage:
    python scripts/train_dpo.py --data-dir ./ --beta 0.1 --epochs 3
    python scripts/train_dpo.py --help
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from src.data.preference_data_generator import PreferenceDataGenerator
from src.train.dpo_config import DPOTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train recommendation LLM with DPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="./", help="Data directory")
    parser.add_argument(
        "--model-name", type=str, 
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--strategy", type=str, default="mixed",
        choices=["model_based", "random", "mixed"],
        help="Negative sampling strategy",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/dpo_checkpoints",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate data and config only, don't train",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DPO TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Generate Preference Data ──
    logger.info("\n📊 Step 1: Generating preference pairs...")

    generator = PreferenceDataGenerator(args.data_dir)
    generator.load_data()

    pairs = generator.generate_pairs(strategy=args.strategy)
    hard_pairs = generator.generate_hard_negatives()

    # Combine model-based and hard negatives
    all_pairs = pairs + hard_pairs
    generator.preference_pairs = all_pairs

    stats = generator.get_statistics()
    print("\n" + "─" * 50)
    print("PREFERENCE DATA STATISTICS")
    print("─" * 50)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key:35s}: {val:.4f}")
        else:
            print(f"  {key:35s}: {val}")

    # Split data
    train_pairs, val_pairs, test_pairs = generator.split_pairs()

    logger.info(
        f"Data split — Train: {len(train_pairs)} | "
        f"Val: {len(val_pairs)} | Test: {len(test_pairs)}"
    )

    # ── Step 2: Configure DPO Training ──
    logger.info("\n⚙️ Step 2: Configuring DPO training...")

    from src.train.dpo_config import LoRAConfig

    config = DPOTrainingConfig(
        model_name=args.model_name,
        beta=args.beta,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        lora=LoRAConfig(rank=args.lora_rank),
    )
    config.validate()

    print("\n" + "─" * 50)
    print("DPO CONFIGURATION")
    print("─" * 50)
    print(f"  Model:           {config.model_name}")
    print(f"  Beta:            {config.beta}")
    print(f"  Learning Rate:   {config.learning_rate}")
    print(f"  Epochs:          {config.num_epochs}")
    print(f"  LoRA Rank:       {config.lora.rank}")
    print(f"  Effective Batch: {config.effective_batch_size}")

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    config.save(os.path.join(args.output_dir, "dpo_config.json"))

    if args.dry_run:
        logger.info("Dry run — skipping model loading and training.")
        # Save preference data for inspection
        df = generator.to_dataframe()
        csv_path = os.path.join(args.output_dir, "preference_pairs.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Preference pairs saved to {csv_path}")
        return

    # ── Step 3: Train with DPO ──
    logger.info("\n🚀 Step 3: Starting DPO training...")

    from src.train.dpo_trainer import RecommendationDPOTrainer

    trainer = RecommendationDPOTrainer(config)
    trainer.initialize()

    # Convert to HuggingFace datasets
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset = Dataset.from_list(val_pairs) if val_pairs else None

    results = trainer.train(train_dataset, eval_dataset)

    print("\n" + "─" * 50)
    print("TRAINING RESULTS")
    print("─" * 50)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:35s}: {val:.4f}")
        else:
            print(f"  {key:35s}: {val}")

    # ── Step 4: Evaluate ──
    logger.info("\n📈 Step 4: Computing rewards...")

    test_dataset = Dataset.from_list(test_pairs) if test_pairs else None
    if test_dataset:
        rewards = trainer.compute_rewards(test_dataset)
        print("\n" + "─" * 50)
        print("REWARD ANALYSIS")
        print("─" * 50)
        print(f"  Reward accuracy:  {rewards['reward_accuracy']:.4f}")
        print(f"  Mean margin:      {float(rewards['margins'].mean()):.4f}")

    print("\n" + "=" * 60)
    print("✅ DPO Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
