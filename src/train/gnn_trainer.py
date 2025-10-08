"""
GNN Training Pipeline with Link Prediction.

Implements the training loop for the GraphSAGE recommender model using
a link prediction objective. The trainer handles:
- Positive/negative edge sampling
- BPR (Bayesian Personalized Ranking) loss
- AUC-ROC evaluation
- Learning rate scheduling with warmup
- Early stopping with patience
- Model checkpointing

The link prediction task trains the model to score existing user-item
edges higher than randomly sampled non-existing edges, learning
meaningful embeddings in the process.

References:
    - Rendle et al., "BPR: Bayesian Personalized Ranking" (2009)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import roc_auc_score

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


class GNNTrainer:
    """
    Training pipeline for GraphSAGE link prediction.

    Handles the complete training lifecycle including data preparation,
    training loop, evaluation, and model checkpointing.

    Attributes:
        model: GraphSAGE model to train.
        device: Computation device.
        optimizer: Parameter optimizer.
        scheduler: Learning rate scheduler.
        history: Training history (loss, metrics per epoch).

    Example:
        >>> trainer = GNNTrainer(model, device="cuda", lr=1e-3)
        >>> trainer.train(graph_data, num_epochs=100, val_edges=val_pos, val_neg=val_neg)
        >>> metrics = trainer.evaluate(graph_data, test_pos, test_neg)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 15,
        checkpoint_dir: str = "models/checkpoints",
    ) -> None:
        """
        Initialize GNN trainer.

        Args:
            model: GraphSAGE model instance.
            device: Device for training ('cuda' or 'cpu').
            lr: Learning rate.
            weight_decay: L2 regularization weight.
            patience: Early stopping patience (epochs without improvement).
            checkpoint_dir: Directory for saving model checkpoints.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_auc": [],
            "learning_rate": [],
        }
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(
            f"GNNTrainer initialized | Device: {self.device} | "
            f"LR: {lr} | Weight decay: {weight_decay} | "
            f"Patience: {patience}"
        )

    def train(
        self,
        graph_data: Any,
        train_edge_index: torch.Tensor,
        val_pos_edges: torch.Tensor,
        val_neg_edges: torch.Tensor,
        num_epochs: int = 100,
        neg_sampling_ratio: float = 1.0,
        log_interval: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Train the GNN model with link prediction objective.

        For each epoch:
        1. Sample negative edges
        2. Compute BPR loss on positive vs negative edges
        3. Evaluate on validation edges
        4. Update learning rate scheduler
        5. Check early stopping

        Args:
            graph_data: PyG Data object with node features and structure.
            train_edge_index: Training edge index for message passing.
            val_pos_edges: Positive validation edges (2, num_val_pos).
            val_neg_edges: Negative validation edges (2, num_val_neg).
            num_epochs: Maximum training epochs.
            neg_sampling_ratio: Ratio of negative to positive samples.
            log_interval: Epochs between detailed logging.

        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        x = graph_data.x.to(self.device)
        train_edges = train_edge_index.to(self.device)
        val_pos = val_pos_edges.to(self.device)
        val_neg = val_neg_edges.to(self.device)

        num_users = graph_data.num_users
        num_items = graph_data.num_items

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Training step
            train_loss = self._train_epoch(
                x=x,
                edge_index=train_edges,
                num_users=num_users,
                num_items=num_items,
                neg_sampling_ratio=neg_sampling_ratio,
            )

            # Validation step
            val_auc = self._evaluate(
                x=x,
                edge_index=train_edges,
                pos_edges=val_pos,
                neg_edges=val_neg,
            )

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_auc)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_auc"].append(val_auc)
            self.history["learning_rate"].append(current_lr)

            epoch_time = time.time() - start_time

            # Logging
            if epoch % log_interval == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )

            # Early stopping check
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_auc, "best_model.pt")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val AUC: {self.best_val_auc:.4f}"
                )
                break

        # Save final model
        self._save_checkpoint(epoch, val_auc, "final_model.pt")

        logger.info(
            f"Training complete | Best val AUC: {self.best_val_auc:.4f} | "
            f"Total epochs: {epoch}"
        )

        return self.history

    def _train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
        neg_sampling_ratio: float,
    ) -> float:
        """
        Execute one training epoch.

        Args:
            x: Node features on device.
            edge_index: Training edges on device.
            num_users: Number of user nodes.
            num_items: Number of item nodes.
            neg_sampling_ratio: Ratio of negative samples.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Encode all nodes
        embeddings = self.model.encode(x, edge_index)

        # Get positive edges (user → item only, forward direction)
        forward_mask = edge_index[0] < num_users
        pos_src = edge_index[0, forward_mask]
        pos_dst = edge_index[1, forward_mask]

        # Sample negative edges
        num_neg = int(pos_src.shape[0] * neg_sampling_ratio)
        neg_src = pos_src[torch.randint(0, pos_src.shape[0], (num_neg,))]
        neg_dst = torch.randint(
            num_users,
            num_users + num_items,
            (num_neg,),
            device=self.device,
        )

        # Compute scores
        pos_scores = self.model.predict_link(embeddings, pos_src, pos_dst)
        neg_scores = self.model.predict_link(embeddings, neg_src, neg_dst)

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        # Sample matching negative for each positive
        min_len = min(pos_scores.shape[0], neg_scores.shape[0])
        bpr_loss = -torch.log(
            torch.sigmoid(pos_scores[:min_len] - neg_scores[:min_len]) + 1e-8
        ).mean()

        # L2 regularization on embeddings
        reg_loss = 0.01 * (
            embeddings[pos_src[:min_len]].norm(2).pow(2)
            + embeddings[pos_dst[:min_len]].norm(2).pow(2)
            + embeddings[neg_dst[:min_len]].norm(2).pow(2)
        ) / min_len

        loss = bpr_loss + reg_loss

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def _evaluate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> float:
        """
        Evaluate model on validation/test edges using AUC-ROC.

        Args:
            x: Node features on device.
            edge_index: Edge index for message passing.
            pos_edges: Positive edges to evaluate.
            neg_edges: Negative edges to evaluate.

        Returns:
            AUC-ROC score.
        """
        self.model.eval()

        embeddings = self.model.encode(x, edge_index)

        pos_scores = self.model.predict_link(
            embeddings, pos_edges[0], pos_edges[1]
        )
        neg_scores = self.model.predict_link(
            embeddings, neg_edges[0], neg_edges[1]
        )

        # Compute AUC-ROC
        scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        labels = np.concatenate([
            np.ones(pos_scores.shape[0]),
            np.zeros(neg_scores.shape[0]),
        ])

        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5  # Default if only one class present

        return float(auc)

    def evaluate(
        self,
        graph_data: Any,
        test_pos_edges: torch.Tensor,
        test_neg_edges: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Full evaluation on test set with multiple metrics.

        Args:
            graph_data: PyG Data object.
            test_pos_edges: Positive test edges.
            test_neg_edges: Negative test edges.
            edge_index: Optional edge index for message passing.
                If None, uses graph_data.edge_index.

        Returns:
            Dictionary with evaluation metrics.
        """
        x = graph_data.x.to(self.device)
        ei = (edge_index if edge_index is not None else graph_data.edge_index).to(self.device)
        pos = test_pos_edges.to(self.device)
        neg = test_neg_edges.to(self.device)

        self.model.eval()

        with torch.no_grad():
            embeddings = self.model.encode(x, ei)

            pos_scores = self.model.predict_link(embeddings, pos[0], pos[1])
            neg_scores = self.model.predict_link(embeddings, neg[0], neg[1])

        # AUC-ROC
        all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        all_labels = np.concatenate([
            np.ones(pos_scores.shape[0]),
            np.zeros(neg_scores.shape[0]),
        ])

        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = 0.5

        # Hit Rate @ K
        hit_rate_10 = self._compute_hit_rate(pos_scores, neg_scores, k=10)
        hit_rate_20 = self._compute_hit_rate(pos_scores, neg_scores, k=20)

        # Mean Reciprocal Rank
        mrr = self._compute_mrr(pos_scores, neg_scores)

        metrics = {
            "auc_roc": auc,
            "hit_rate_10": hit_rate_10,
            "hit_rate_20": hit_rate_20,
            "mrr": mrr,
            "mean_pos_score": float(pos_scores.mean()),
            "mean_neg_score": float(neg_scores.mean()),
            "score_gap": float(pos_scores.mean() - neg_scores.mean()),
        }

        logger.info(f"Test evaluation: {metrics}")
        return metrics

    @staticmethod
    def _compute_hit_rate(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        k: int,
    ) -> float:
        """
        Compute Hit Rate @ K.

        For each positive edge, checks if its score ranks in the top K
        among all negative edges.
        """
        hits = 0
        for pos_score in pos_scores:
            # Count how many negatives score higher
            rank = (neg_scores >= pos_score).sum().item() + 1
            if rank <= k:
                hits += 1
        return hits / max(len(pos_scores), 1)

    @staticmethod
    def _compute_mrr(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        For each positive edge, computes 1/rank where rank is the
        position among all (positive + negative) scores.
        """
        reciprocal_ranks = []
        for pos_score in pos_scores:
            rank = (neg_scores >= pos_score).sum().item() + 1
            reciprocal_ranks.append(1.0 / rank)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def _save_checkpoint(
        self,
        epoch: int,
        val_auc: float,
        filename: str,
    ) -> None:
        """Save model checkpoint to disk."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_auc": val_auc,
                "history": self.history,
            },
            path,
        )
        logger.debug(f"Checkpoint saved: {path} (AUC: {val_auc:.4f})")

    def load_checkpoint(self, filename: str = "best_model.pt") -> None:
        """
        Load a saved model checkpoint.

        Args:
            filename: Checkpoint filename to load.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_auc = checkpoint.get("val_auc", 0.0)
        self.history = checkpoint.get("history", self.history)

        logger.info(
            f"Loaded checkpoint: {path} | "
            f"Epoch: {checkpoint.get('epoch', '?')} | "
            f"AUC: {self.best_val_auc:.4f}"
        )

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training run."""
        return {
            "total_epochs": len(self.history["train_loss"]),
            "best_val_auc": self.best_val_auc,
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_auc": self.history["val_auc"][-1] if self.history["val_auc"] else None,
            "final_lr": self.history["learning_rate"][-1] if self.history["learning_rate"] else None,
        }
