#!/usr/bin/env python3
"""
GNN Training Entry Point.

Trains the GraphSAGE recommender model on the Amazon Prime Pantry
interaction graph using link prediction objective.

Usage:
    python scripts/train_gnn.py --data-dir ./ --epochs 100 --lr 0.001
    python scripts/train_gnn.py --help
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from loguru import logger

from src.data.dataset_loader import AmazonDatasetLoader
from src.graph.graph_builder import InteractionGraphBuilder
from src.graph.gnn_model import GraphSAGERecommender
from src.train.gnn_trainer import GNNTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE recommender on Amazon dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default="./",
        help="Path to data directory containing final_dataset.xlsx",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128,
        help="Node feature embedding dimensionality",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden layer dimensionality",
    )
    parser.add_argument(
        "--output-dim", type=int, default=128,
        help="Output embedding dimensionality",
    )
    parser.add_argument(
        "--num-layers", type=int, default=3,
        help="Number of GraphSAGE layers (hop depth)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5,
        help="L2 regularization weight",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="models/checkpoints",
        help="Directory for model checkpoints",
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # ── Step 1: Load Data ──
    logger.info("=" * 60)
    logger.info("Step 1: Loading dataset...")
    logger.info("=" * 60)

    loader = AmazonDatasetLoader(args.data_dir)
    interactions = loader.load_data()

    summary = loader.get_summary()
    for key, val in summary.items():
        logger.info(f"  {key}: {val}")

    # ── Step 2: Build Graph ──
    logger.info("=" * 60)
    logger.info("Step 2: Building interaction graph...")
    logger.info("=" * 60)

    builder = InteractionGraphBuilder(
        interactions_df=interactions,
        num_users=loader.num_users,
        num_items=loader.num_items,
    )

    graph_data = builder.build_graph(embedding_dim=args.embedding_dim)

    stats = builder.get_graph_statistics()
    logger.info("Graph statistics:")
    for key, val in stats.items():
        logger.info(f"  {key}: {val}")

    # ── Step 3: Split Edges ──
    logger.info("=" * 60)
    logger.info("Step 3: Splitting edges for training...")
    logger.info("=" * 60)

    train_edge_index, test_pos, test_neg = builder.get_train_test_edges(
        test_ratio=0.1
    )

    # Create validation edges from a portion of test
    val_size = test_pos.shape[1] // 2
    val_pos = test_pos[:, :val_size]
    val_neg = test_neg[:, :val_size]
    test_pos_final = test_pos[:, val_size:]
    test_neg_final = test_neg[:, val_size:]

    # ── Step 4: Initialize Model ──
    logger.info("=" * 60)
    logger.info("Step 4: Initializing GraphSAGE model...")
    logger.info("=" * 60)

    model = GraphSAGERecommender(
        in_channels=args.embedding_dim,
        hidden_channels=args.hidden_dim,
        out_channels=args.output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── Step 5: Train ──
    logger.info("=" * 60)
    logger.info("Step 5: Training GNN...")
    logger.info("=" * 60)

    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )

    history = trainer.train(
        graph_data=graph_data,
        train_edge_index=train_edge_index,
        val_pos_edges=val_pos,
        val_neg_edges=val_neg,
        num_epochs=args.epochs,
    )

    # ── Step 6: Evaluate ──
    logger.info("=" * 60)
    logger.info("Step 6: Final evaluation...")
    logger.info("=" * 60)

    # Load best model
    trainer.load_checkpoint("best_model.pt")

    metrics = trainer.evaluate(
        graph_data=graph_data,
        test_pos_edges=test_pos_final,
        test_neg_edges=test_neg_final,
        edge_index=train_edge_index,
    )

    logger.info("Final test metrics:")
    for key, val in metrics.items():
        logger.info(f"  {key}: {val:.4f}")

    training_summary = trainer.get_training_summary()
    logger.info(f"Training summary: {training_summary}")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
