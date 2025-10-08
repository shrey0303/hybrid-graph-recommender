#!/usr/bin/env python3
"""
Evaluation and Graph Analysis Script.

Loads the dataset, builds the interaction graph, and prints detailed
statistics about the graph structure. Optionally loads a trained model
and evaluates it on the test set.

Usage:
    python scripts/run_evaluation.py --data-dir ./
    python scripts/run_evaluation.py --data-dir ./ --model-path models/checkpoints/best_model.pt
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from loguru import logger

from src.data.dataset_loader import AmazonDatasetLoader
from src.graph.graph_builder import InteractionGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation model and analyze graph",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./",
        help="Path to data directory",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128,
        help="Node embedding dimensionality",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model checkpoint (optional)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save results as JSON (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RECOMMENDATION SYSTEM — EVALUATION & ANALYSIS")
    logger.info("=" * 60)

    # Load data
    logger.info("\n📊 Loading dataset...")
    loader = AmazonDatasetLoader(args.data_dir)
    interactions = loader.load_data()
    data_summary = loader.get_summary()

    print("\n" + "─" * 50)
    print("DATASET SUMMARY")
    print("─" * 50)
    for key, val in data_summary.items():
        if isinstance(val, float):
            print(f"  {key:35s}: {val:.6f}")
        else:
            print(f"  {key:35s}: {val}")

    # Build graph
    logger.info("\n🔗 Building interaction graph...")
    builder = InteractionGraphBuilder(
        interactions_df=interactions,
        num_users=loader.num_users,
        num_items=loader.num_items,
    )
    graph = builder.build_graph(embedding_dim=args.embedding_dim)
    graph_stats = builder.get_graph_statistics()

    print("\n" + "─" * 50)
    print("GRAPH STATISTICS")
    print("─" * 50)
    for key, val in graph_stats.items():
        if isinstance(val, float):
            print(f"  {key:35s}: {val:.6f}")
        else:
            print(f"  {key:35s}: {val}")

    # Train/test split analysis
    logger.info("\n✂️ Analyzing train/test split...")
    train_ei, test_pos, test_neg = builder.get_train_test_edges(test_ratio=0.1)

    print("\n" + "─" * 50)
    print("EDGE SPLIT ANALYSIS")
    print("─" * 50)
    print(f"  {'Training edges':35s}: {train_ei.shape[1]}")
    print(f"  {'Test positive edges':35s}: {test_pos.shape[1]}")
    print(f"  {'Test negative edges':35s}: {test_neg.shape[1]}")

    # Data split analysis
    train_df, val_df, test_df = loader.split_data()

    print("\n" + "─" * 50)
    print("DATA SPLIT (Temporal)")
    print("─" * 50)
    print(f"  {'Training interactions':35s}: {len(train_df)}")
    print(f"  {'Validation interactions':35s}: {len(val_df)}")
    print(f"  {'Test interactions':35s}: {len(test_df)}")

    # Model evaluation (if checkpoint provided)
    if args.model_path and os.path.isfile(args.model_path):
        logger.info(f"\n🤖 Loading model from {args.model_path}...")
        from src.graph.gnn_model import GraphSAGERecommender
        from src.train.gnn_trainer import GNNTrainer

        # Try to load checkpoint to get model config
        checkpoint = torch.load(args.model_path, map_location="cpu")

        model = GraphSAGERecommender(
            in_channels=args.embedding_dim,
            hidden_channels=256,
            out_channels=128,
            num_layers=3,
        )

        trainer = GNNTrainer(model=model, device="cpu")
        trainer.load_checkpoint(os.path.basename(args.model_path))

        metrics = trainer.evaluate(
            graph_data=graph,
            test_pos_edges=test_pos,
            test_neg_edges=test_neg,
        )

        print("\n" + "─" * 50)
        print("MODEL EVALUATION METRICS")
        print("─" * 50)
        for key, val in metrics.items():
            print(f"  {key:35s}: {val:.4f}")
    else:
        logger.info(
            "\nℹ️  No model checkpoint provided. "
            "Train first with: python scripts/train_gnn.py"
        )

    # Save results to JSON
    if args.output_json:
        results = {
            "dataset": data_summary,
            "graph": graph_stats,
            "edge_split": {
                "train": train_ei.shape[1],
                "test_pos": test_pos.shape[1],
                "test_neg": test_neg.shape[1],
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output_json}")

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
