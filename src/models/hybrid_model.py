"""
Hybrid GNN-LLM Fusion Model.

Implements input-level fusion between Graph Neural Network embeddings and
Large Language Model embeddings for hybrid recommendation. Uses a gated
fusion mechanism that learns to dynamically weight the contribution of
graph structure vs. text semantics for each prediction.

Architecture:
    GNN Embeddings ──→ Projection ──┐
                                     ├──→ Gated Fusion ──→ MLP ──→ Output
    LLM Embeddings ──→ Projection ──┘

The gated fusion mechanism learns a soft attention weight α ∈ [0, 1]
that controls the mixing ratio: output = α · GNN + (1 - α) · LLM.
This allows the model to adaptively rely more on graph structure for
users with rich interaction histories, and more on text semantics for
cold-start scenarios.

References:
    - "Combining Graph Neural Networks with Language Models for Recommendation" (2023)
    - Gating mechanism inspired by GRU/LSTM gating
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class GatedFusionLayer(nn.Module):
    """
    Gated fusion layer for combining two embedding sources.

    Learns a data-dependent gate that controls the mixing ratio
    between two input embedding streams.

    Attributes:
        projection_dim: Common projection dimensionality.
        gate: Linear layer computing fusion gate values.
    """

    def __init__(self, input_dim_a: int, input_dim_b: int, projection_dim: int) -> None:
        """
        Initialize gated fusion.

        Args:
            input_dim_a: Dimensionality of first input (e.g., GNN embeddings).
            input_dim_b: Dimensionality of second input (e.g., LLM embeddings).
            projection_dim: Common space dimensionality for fusion.
        """
        super().__init__()

        self.proj_a = nn.Linear(input_dim_a, projection_dim)
        self.proj_b = nn.Linear(input_dim_b, projection_dim)

        # Gate network: takes concatenated projections → scalar gate
        self.gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, 1),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse two embedding sources with learned gating.

        Args:
            emb_a: First source embeddings (batch_size, input_dim_a).
            emb_b: Second source embeddings (batch_size, input_dim_b).

        Returns:
            Tuple of (fused_embedding, gate_values):
                - fused_embedding: (batch_size, projection_dim)
                - gate_values: (batch_size, 1) — α values for analysis
        """
        # Project to common space
        proj_a = self.proj_a(emb_a)
        proj_b = self.proj_b(emb_b)

        # Compute gate value
        concat = torch.cat([proj_a, proj_b], dim=-1)
        alpha = self.gate(concat)  # (batch_size, 1)

        # Gated fusion: α * GNN + (1 - α) * LLM
        fused = alpha * proj_a + (1 - alpha) * proj_b

        # Layer normalization for training stability
        fused = self.layer_norm(fused)

        return fused, alpha


class HybridGNN_LLM(nn.Module):
    """
    Hybrid model fusing GNN graph embeddings with LLM text embeddings.

    Combines structural signals from the user-item interaction graph
    (via GraphSAGE) with semantic signals from product descriptions
    (via LLM). Uses gated fusion for adaptive weighting and an MLP
    head for downstream prediction.

    Attributes:
        gnn_dim: GNN embedding dimensionality.
        llm_dim: LLM embedding dimensionality.
        fusion_dim: Fused representation dimensionality.
        output_dim: Final output dimensionality.

    Example:
        >>> hybrid = HybridGNN_LLM(
        ...     gnn_dim=128, llm_dim=768,
        ...     fusion_dim=256, output_dim=128,
        ... )
        >>> output = hybrid(gnn_emb, llm_emb)
        >>> print(output.shape)  # (batch_size, 128)
    """

    def __init__(
        self,
        gnn_dim: int,
        llm_dim: int,
        fusion_dim: int = 256,
        output_dim: int = 128,
        num_mlp_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize hybrid GNN-LLM model.

        Args:
            gnn_dim: Dimensionality of GNN node embeddings.
            llm_dim: Dimensionality of LLM text embeddings.
            fusion_dim: Hidden dimensionality after fusion.
            output_dim: Final output embedding dimensionality.
            num_mlp_layers: Number of MLP layers after fusion.
            dropout: Dropout probability in MLP layers.
        """
        super().__init__()

        self.gnn_dim = gnn_dim
        self.llm_dim = llm_dim
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim

        # Gated fusion layer
        self.fusion = GatedFusionLayer(
            input_dim_a=gnn_dim,
            input_dim_b=llm_dim,
            projection_dim=fusion_dim,
        )

        # MLP prediction head
        mlp_layers = []
        current_dim = fusion_dim

        for i in range(num_mlp_layers - 1):
            next_dim = max(output_dim, current_dim // 2)
            mlp_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim

        mlp_layers.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*mlp_layers)

        # Initialize weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"HybridGNN_LLM initialized | "
            f"GNN: {gnn_dim}d | LLM: {llm_dim}d | "
            f"Fusion: {fusion_dim}d | Output: {output_dim}d | "
            f"Params: {total_params:,} ({trainable_params:,} trainable)"
        )

    def _init_weights(self) -> None:
        """Initialize model weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        gnn_embeddings: torch.Tensor,
        llm_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: fuse GNN and LLM embeddings, produce output.

        Args:
            gnn_embeddings: Graph embeddings (batch_size, gnn_dim).
            llm_embeddings: Text embeddings (batch_size, llm_dim).

        Returns:
            Fused output embeddings of shape (batch_size, output_dim).
        """
        # Gated fusion
        fused, _ = self.fusion(gnn_embeddings, llm_embeddings)

        # MLP prediction head
        output = self.mlp(fused)

        return output

    def forward_with_gate(
        self,
        gnn_embeddings: torch.Tensor,
        llm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns gate values for analysis.

        Args:
            gnn_embeddings: Graph embeddings (batch_size, gnn_dim).
            llm_embeddings: Text embeddings (batch_size, llm_dim).

        Returns:
            Tuple of (output, gate_values):
                - output: (batch_size, output_dim)
                - gate_values: (batch_size, 1) — how much GNN is weighted
        """
        fused, gate_values = self.fusion(gnn_embeddings, llm_embeddings)
        output = self.mlp(fused)
        return output, gate_values

    def get_fusion_weights(
        self,
        gnn_embeddings: torch.Tensor,
        llm_embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Analyze how the model weights GNN vs LLM for a batch.

        Useful for interpretability — understanding when the model
        relies more on graph structure vs. text semantics.

        Args:
            gnn_embeddings: Graph embeddings (batch_size, gnn_dim).
            llm_embeddings: Text embeddings (batch_size, llm_dim).

        Returns:
            Dictionary with mean/std of gate values.
        """
        with torch.no_grad():
            _, gate_values = self.fusion(gnn_embeddings, llm_embeddings)

        return {
            "gnn_weight_mean": float(gate_values.mean()),
            "gnn_weight_std": float(gate_values.std()),
            "llm_weight_mean": float(1 - gate_values.mean()),
            "llm_weight_std": float(gate_values.std()),
        }

    def __repr__(self) -> str:
        return (
            f"HybridGNN_LLM("
            f"gnn={self.gnn_dim}d, "
            f"llm={self.llm_dim}d, "
            f"fusion={self.fusion_dim}d, "
            f"output={self.output_dim}d)"
        )
