"""
GraphSAGE Recommendation Model.

Implements a multi-layer GraphSAGE architecture for learning node embeddings
in the user-item interaction graph. Supports configurable depth, batch
normalization, and dropout for production-grade recommendation.

Architecture:
    Input → [SAGEConv → BatchNorm → ReLU → Dropout] × L → Output Embeddings

The model learns to aggregate information from multi-hop neighborhoods,
capturing both local and global graph structure. This allows users to be
represented by the items their neighbors purchased, and items by the users
who purchased similar products.

References:
    - Hamilton et al., "Inductive Representation Learning on Large Graphs" (2017)
    - GraphSAGE: https://arxiv.org/abs/1706.02216
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    SAGEConv = None
    logger.warning(
        "torch_geometric not installed. GraphSAGE model unavailable. "
        "Install with: pip install torch-geometric"
    )


class GraphSAGERecommender(nn.Module):
    """
    GraphSAGE-based recommender for learning user/item embeddings.

    Uses multi-layer GraphSAGE convolutions with batch normalization
    and dropout to produce rich node representations from the
    user-item interaction graph.

    Attributes:
        in_channels: Input feature dimensionality.
        hidden_channels: Hidden layer dimensionality.
        out_channels: Output embedding dimensionality.
        num_layers: Number of GraphSAGE convolution layers.
        dropout: Dropout probability.

    Example:
        >>> model = GraphSAGERecommender(
        ...     in_channels=128,
        ...     hidden_channels=256,
        ...     out_channels=128,
        ...     num_layers=3,
        ...     dropout=0.2,
        ... )
        >>> embeddings = model.encode(graph.x, graph.edge_index)
        >>> scores = model.predict_link(embeddings, user_idx, item_idx)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize GraphSAGE recommender.

        Args:
            in_channels: Number of input features per node.
            hidden_channels: Number of hidden units in intermediate layers.
            out_channels: Dimensionality of output embeddings.
            num_layers: Number of SAGEConv layers (controls hop depth).
            dropout: Dropout rate between layers.
            normalize_embeddings: Whether to L2-normalize output embeddings.

        Raises:
            ImportError: If torch_geometric is not installed.
            ValueError: If num_layers < 1 or dimensions are invalid.
        """
        super().__init__()

        if SAGEConv is None:
            raise ImportError(
                "torch_geometric is required. "
                "Install with: pip install torch-geometric"
            )

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if in_channels <= 0 or hidden_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"Channel dimensions must be positive. Got "
                f"in={in_channels}, hidden={hidden_channels}, out={out_channels}"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings

        # Build GraphSAGE convolution layers
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: in_channels → hidden_channels
        self.convolutions.append(
            SAGEConv(in_channels, hidden_channels, aggr="mean")
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers: hidden_channels → hidden_channels
        for _ in range(num_layers - 2):
            self.convolutions.append(
                SAGEConv(hidden_channels, hidden_channels, aggr="mean")
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer: hidden_channels → out_channels
        if num_layers > 1:
            self.convolutions.append(
                SAGEConv(hidden_channels, out_channels, aggr="mean")
            )
            self.batch_norms.append(nn.BatchNorm1d(out_channels))

        # Output projection for single-layer case
        if num_layers == 1:
            self.convolutions[0] = SAGEConv(in_channels, out_channels, aggr="mean")
            self.batch_norms[0] = nn.BatchNorm1d(out_channels)

        logger.info(
            f"GraphSAGERecommender initialized | "
            f"Layers: {num_layers} | "
            f"Dims: {in_channels} → {hidden_channels} → {out_channels} | "
            f"Dropout: {dropout}"
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate node embeddings through multi-layer message passing.

        Performs L rounds of neighborhood aggregation, where each round
        collects and aggregates features from 1-hop neighbors.
        After L rounds, each node's embedding captures information
        from its L-hop neighborhood.

        Args:
            x: Node feature matrix of shape (num_nodes, in_channels).
            edge_index: Edge index of shape (2, num_edges).

        Returns:
            Node embeddings of shape (num_nodes, out_channels).
        """
        for i, (conv, bn) in enumerate(zip(self.convolutions, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)

            # Apply activation and dropout for all but the last layer
            if i < len(self.convolutions) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # L2 normalize embeddings for stable dot-product scoring
        if self.normalize_embeddings:
            x = F.normalize(x, p=2, dim=-1)

        return x

    def predict_link(
        self,
        embeddings: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict link scores between source and destination node pairs.

        Uses dot-product similarity between node embeddings as the
        link prediction score. Higher scores indicate more likely edges.

        Args:
            embeddings: Full node embedding matrix (num_nodes, out_channels).
            src_indices: Source node indices of shape (num_pairs,).
            dst_indices: Destination node indices of shape (num_pairs,).

        Returns:
            Link prediction scores of shape (num_pairs,).
        """
        src_emb = embeddings[src_indices]
        dst_emb = embeddings[dst_indices]

        # Dot product similarity
        scores = (src_emb * dst_emb).sum(dim=-1)
        return scores

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        src_indices: Optional[torch.Tensor] = None,
        dst_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full forward pass: encode + optional link prediction.

        Args:
            x: Node features of shape (num_nodes, in_channels).
            edge_index: Edge index of shape (2, num_edges).
            src_indices: Optional source indices for link prediction.
            dst_indices: Optional destination indices for link prediction.

        Returns:
            Tuple of (embeddings, scores). Scores is None if indices
            are not provided.
        """
        embeddings = self.encode(x, edge_index)

        scores = None
        if src_indices is not None and dst_indices is not None:
            scores = self.predict_link(embeddings, src_indices, dst_indices)

        return embeddings, scores

    def get_user_embeddings(
        self,
        embeddings: torch.Tensor,
        num_users: int,
    ) -> torch.Tensor:
        """
        Extract user embeddings from the full embedding matrix.

        Args:
            embeddings: Full node embeddings (num_nodes, out_channels).
            num_users: Number of user nodes (first N nodes are users).

        Returns:
            User embeddings of shape (num_users, out_channels).
        """
        return embeddings[:num_users]

    def get_item_embeddings(
        self,
        embeddings: torch.Tensor,
        num_users: int,
    ) -> torch.Tensor:
        """
        Extract item embeddings from the full embedding matrix.

        Args:
            embeddings: Full node embeddings (num_nodes, out_channels).
            num_users: Number of user nodes (items start after users).

        Returns:
            Item embeddings of shape (num_items, out_channels).
        """
        return embeddings[num_users:]

    def recommend_items(
        self,
        embeddings: torch.Tensor,
        user_idx: int,
        num_users: int,
        top_k: int = 10,
        exclude_items: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate top-K item recommendations for a given user.

        Computes dot-product scores between the user embedding and all
        item embeddings, then returns the top-K items by score.

        Args:
            embeddings: Full node embeddings (num_nodes, out_channels).
            user_idx: Index of the user to recommend for.
            num_users: Number of user nodes.
            top_k: Number of recommendations to return.
            exclude_items: Optional tensor of item indices to exclude
                (e.g., already purchased items).

        Returns:
            Tuple of (item_indices, scores) for top-K recommendations.
        """
        user_emb = embeddings[user_idx].unsqueeze(0)  # (1, out_channels)
        item_embs = self.get_item_embeddings(embeddings, num_users)  # (num_items, out_channels)

        # Compute scores via dot product
        scores = torch.matmul(item_embs, user_emb.t()).squeeze(-1)

        # Exclude already-interacted items
        if exclude_items is not None:
            scores[exclude_items] = float("-inf")

        # Get top-K
        top_k = min(top_k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, top_k)

        return top_indices, top_scores

    def __repr__(self) -> str:
        return (
            f"GraphSAGERecommender("
            f"in={self.in_channels}, "
            f"hidden={self.hidden_channels}, "
            f"out={self.out_channels}, "
            f"layers={self.num_layers}, "
            f"dropout={self.dropout})"
        )
