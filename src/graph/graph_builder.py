"""
Interaction Graph Builder for GNN-based Recommendation.

Constructs a bipartite user-item interaction graph from purchase/review data
using PyTorch Geometric's Data format. The graph captures the structural
relationships between users and items that sequential LLM models miss.

Architecture:
    - Bipartite graph: User nodes ↔ Item nodes
    - Edges: User-Item interactions (purchases/reviews)
    - Bidirectional: Edges in both directions for message passing
    - Node features: Learnable embeddings initialized from interaction statistics

References:
    - Hamilton et al., "Inductive Representation Learning on Large Graphs" (2017)
    - Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (2018)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None
    logger.warning(
        "torch_geometric not installed. Install with: "
        "pip install torch-geometric"
    )


class InteractionGraphBuilder:
    """
    Build a bipartite user-item interaction graph for GNN-based recommendation.

    Constructs a heterogeneous graph where users and items are nodes,
    connected by purchase/review edges. The graph is stored in PyTorch
    Geometric's Data format for seamless integration with GNN layers.

    Attributes:
        interactions_df: Source interaction data.
        num_users: Number of unique users.
        num_items: Number of unique items.
        graph_data: Constructed PyG Data object.

    Example:
        >>> builder = InteractionGraphBuilder(interactions_df, num_users=1000, num_items=500)
        >>> graph = builder.build_graph(embedding_dim=128)
        >>> print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
        >>> stats = builder.get_graph_statistics()
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
    ) -> None:
        """
        Initialize the graph builder.

        Args:
            interactions_df: DataFrame with user-item interactions.
                Must contain columns specified by user_col and item_col.
            num_users: Total number of unique users.
            num_items: Total number of unique items.
            user_col: Column name for user integer indices.
            item_col: Column name for item integer indices.

        Raises:
            ValueError: If required columns are missing or counts are invalid.
            ImportError: If torch_geometric is not installed.
        """
        if Data is None:
            raise ImportError(
                "torch_geometric is required for graph construction. "
                "Install with: pip install torch-geometric"
            )

        if user_col not in interactions_df.columns:
            raise ValueError(
                f"Column '{user_col}' not found in DataFrame. "
                f"Available: {list(interactions_df.columns)}"
            )
        if item_col not in interactions_df.columns:
            raise ValueError(
                f"Column '{item_col}' not found in DataFrame. "
                f"Available: {list(interactions_df.columns)}"
            )

        if num_users <= 0 or num_items <= 0:
            raise ValueError(
                f"num_users and num_items must be positive. "
                f"Got num_users={num_users}, num_items={num_items}"
            )

        self.interactions_df = interactions_df.copy()
        self.num_users = num_users
        self.num_items = num_items
        self.user_col = user_col
        self.item_col = item_col
        self.graph_data: Optional[Data] = None

        logger.info(
            f"InteractionGraphBuilder initialized | "
            f"Users: {num_users} | Items: {num_items} | "
            f"Interactions: {len(interactions_df)}"
        )

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (users + items) in the graph."""
        return self.num_users + self.num_items

    def build_graph(
        self,
        embedding_dim: int = 128,
        use_degree_features: bool = True,
    ) -> Data:
        """
        Construct the bipartite user-item interaction graph.

        Creates a PyG Data object with:
        - Bidirectional edges between users and items
        - Node features initialized from interaction statistics
        - Node type labels for heterogeneous processing

        Args:
            embedding_dim: Dimensionality of node feature embeddings.
            use_degree_features: Whether to incorporate degree-based features.

        Returns:
            PyG Data object containing the constructed graph.

        Raises:
            ValueError: If no valid edges can be constructed.
        """
        logger.info("Building interaction graph...")

        # Step 1: Construct edge index
        edge_index = self._build_edge_index()
        logger.info(f"Edge index shape: {edge_index.shape}")

        # Step 2: Generate node features
        node_features = self._generate_node_features(
            embedding_dim=embedding_dim,
            use_degree_features=use_degree_features,
            edge_index=edge_index,
        )
        logger.info(f"Node features shape: {node_features.shape}")

        # Step 3: Create node type labels (0 = user, 1 = item)
        node_types = torch.cat([
            torch.zeros(self.num_users, dtype=torch.long),
            torch.ones(self.num_items, dtype=torch.long),
        ])

        # Step 4: Assemble PyG Data object
        self.graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=self.num_nodes,
            node_type=node_types,
            num_users=self.num_users,
            num_items=self.num_items,
        )

        logger.info(
            f"Graph built successfully | "
            f"Nodes: {self.graph_data.num_nodes} | "
            f"Edges: {self.graph_data.num_edges}"
        )

        return self.graph_data

    def _build_edge_index(self) -> torch.Tensor:
        """
        Construct bidirectional edge index from interaction data.

        Item node indices are offset by num_users to create a unified
        node index space: [0, num_users) for users, [num_users, num_users + num_items)
        for items.

        Returns:
            Edge index tensor of shape (2, num_edges * 2) representing
            bidirectional connections.

        Raises:
            ValueError: If no valid edges can be constructed.
        """
        user_indices = self.interactions_df[self.user_col].values.astype(np.int64)
        item_indices = self.interactions_df[self.item_col].values.astype(np.int64)

        # Validate index ranges
        if len(user_indices) == 0:
            raise ValueError("No interactions provided for edge construction.")

        max_user_idx = user_indices.max()
        max_item_idx = item_indices.max()

        if max_user_idx >= self.num_users:
            raise ValueError(
                f"User index {max_user_idx} exceeds num_users={self.num_users}. "
                "Ensure ID mappings are consistent."
            )
        if max_item_idx >= self.num_items:
            raise ValueError(
                f"Item index {max_item_idx} exceeds num_items={self.num_items}. "
                "Ensure ID mappings are consistent."
            )

        # Offset item indices so they don't overlap with user indices
        item_indices_offset = item_indices + self.num_users

        # Forward edges: user → item
        forward_src = torch.tensor(user_indices, dtype=torch.long)
        forward_dst = torch.tensor(item_indices_offset, dtype=torch.long)

        # Reverse edges: item → user (for bidirectional message passing)
        reverse_src = forward_dst.clone()
        reverse_dst = forward_src.clone()

        # Combine: (forward + reverse)
        edge_src = torch.cat([forward_src, reverse_src], dim=0)
        edge_dst = torch.cat([forward_dst, reverse_dst], dim=0)

        # Remove duplicate edges (same user-item pair appearing multiple times)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def _generate_node_features(
        self,
        embedding_dim: int,
        use_degree_features: bool,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate initial node feature vectors.

        Combines:
        1. Learnable random embeddings (initialized with Xavier uniform)
        2. Degree-based features (normalized node degree as additional signal)

        Args:
            embedding_dim: Target dimensionality for node features.
            use_degree_features: Whether to include degree features.
            edge_index: Constructed edge index for degree computation.

        Returns:
            Node feature tensor of shape (num_nodes, embedding_dim).
        """
        num_nodes = self.num_nodes

        # Base features: Xavier-initialized embeddings
        features = torch.empty(num_nodes, embedding_dim)
        torch.nn.init.xavier_uniform_(features)

        if use_degree_features:
            # Compute node degrees from edge index
            degrees = torch.zeros(num_nodes, dtype=torch.float32)

            # In-degree from the destination side
            if edge_index.shape[1] > 0:
                src_nodes = edge_index[0]
                for node_idx in src_nodes:
                    degrees[node_idx] += 1

            # Normalize degrees to [0, 1] for stability
            max_degree = degrees.max()
            if max_degree > 0:
                normalized_degrees = degrees / max_degree
            else:
                normalized_degrees = degrees

            # Inject degree information into the first feature dimension
            features[:, 0] = normalized_degrees

            # Add node type signal to second feature dimension
            features[:self.num_users, 1] = 1.0   # User indicator
            features[self.num_users:, 1] = -1.0   # Item indicator

        return features

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Compute and return comprehensive graph statistics.

        Returns:
            Dictionary containing:
            - num_nodes: Total node count
            - num_user_nodes: Number of user nodes
            - num_item_nodes: Number of item nodes
            - num_edges: Total (bidirectional) edge count
            - num_interactions: Original interaction count
            - avg_degree: Average node degree
            - user_avg_degree: Average degree for user nodes
            - item_avg_degree: Average degree for item nodes
            - density: Graph density (edges / possible_edges)

        Raises:
            RuntimeError: If graph hasn't been built yet.
        """
        if self.graph_data is None:
            raise RuntimeError(
                "Graph not built yet. Call build_graph() first."
            )

        edge_index = self.graph_data.edge_index
        num_edges = edge_index.shape[1]
        num_nodes = self.graph_data.num_nodes

        # Compute degree distributions
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(num_edges):
            src = edge_index[0, i].item()
            degrees[src] += 1

        user_degrees = degrees[:self.num_users].float()
        item_degrees = degrees[self.num_users:].float()

        # Graph density (for bipartite: edges / (users * items * 2))
        max_possible_edges = 2 * self.num_users * self.num_items
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        stats = {
            "num_nodes": num_nodes,
            "num_user_nodes": self.num_users,
            "num_item_nodes": self.num_items,
            "num_edges": num_edges,
            "num_interactions": len(self.interactions_df),
            "avg_degree": float(degrees.float().mean()),
            "user_avg_degree": float(user_degrees.mean()),
            "item_avg_degree": float(item_degrees.mean()),
            "user_max_degree": int(user_degrees.max()) if len(user_degrees) > 0 else 0,
            "item_max_degree": int(item_degrees.max()) if len(item_degrees) > 0 else 0,
            "density": density,
        }

        logger.info(f"Graph statistics: {stats}")
        return stats

    def get_train_test_edges(
        self,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split edges into train/test sets for link prediction evaluation.

        Removes a fraction of edges for testing and generates negative
        samples (non-existing edges) for evaluation.

        Args:
            test_ratio: Fraction of edges to hold out for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (train_edge_index, test_pos_edges, test_neg_edges).

        Raises:
            RuntimeError: If graph hasn't been built yet.
        """
        if self.graph_data is None:
            raise RuntimeError("Graph not built yet. Call build_graph() first.")

        rng = np.random.RandomState(random_state)
        edge_index = self.graph_data.edge_index

        # Only consider forward edges (user → item) for splitting
        forward_mask = edge_index[0] < self.num_users
        forward_edges = edge_index[:, forward_mask]

        num_forward = forward_edges.shape[1]
        num_test = max(1, int(num_forward * test_ratio))
        num_train = num_forward - num_test

        # Shuffle and split
        perm = rng.permutation(num_forward)
        train_perm = perm[:num_train]
        test_perm = perm[num_train:]

        train_forward = forward_edges[:, train_perm]
        test_pos = forward_edges[:, test_perm]

        # Add reverse edges back for training
        train_reverse = torch.stack([train_forward[1], train_forward[0]], dim=0)
        train_edge_index = torch.cat([train_forward, train_reverse], dim=1)

        # Generate negative samples for testing
        test_neg = self._sample_negative_edges(
            num_samples=test_pos.shape[1],
            existing_edges=edge_index,
            rng=rng,
        )

        logger.info(
            f"Edge split — Train: {train_edge_index.shape[1]} | "
            f"Test pos: {test_pos.shape[1]} | Test neg: {test_neg.shape[1]}"
        )

        return train_edge_index, test_pos, test_neg

    def _sample_negative_edges(
        self,
        num_samples: int,
        existing_edges: torch.Tensor,
        rng: np.random.RandomState,
    ) -> torch.Tensor:
        """
        Sample negative edges (non-existing user-item pairs).

        Args:
            num_samples: Number of negative edges to generate.
            existing_edges: Current edge index to avoid sampling existing edges.
            rng: Random number generator for reproducibility.

        Returns:
            Negative edge index of shape (2, num_samples).
        """
        # Build set of existing edges for fast lookup
        existing_set = set()
        for i in range(existing_edges.shape[1]):
            src, dst = existing_edges[0, i].item(), existing_edges[1, i].item()
            existing_set.add((src, dst))

        neg_src = []
        neg_dst = []
        attempts = 0
        max_attempts = num_samples * 10

        while len(neg_src) < num_samples and attempts < max_attempts:
            u = rng.randint(0, self.num_users)
            i = rng.randint(0, self.num_items) + self.num_users

            if (u, i) not in existing_set:
                neg_src.append(u)
                neg_dst.append(i)
                existing_set.add((u, i))

            attempts += 1

        if len(neg_src) < num_samples:
            logger.warning(
                f"Could only generate {len(neg_src)}/{num_samples} "
                f"negative samples after {max_attempts} attempts."
            )

        return torch.tensor([neg_src, neg_dst], dtype=torch.long)

    def __repr__(self) -> str:
        if self.graph_data is not None:
            return (
                f"InteractionGraphBuilder("
                f"nodes={self.num_nodes}, "
                f"edges={self.graph_data.num_edges}, "
                f"users={self.num_users}, "
                f"items={self.num_items})"
            )
        return (
            f"InteractionGraphBuilder("
            f"users={self.num_users}, "
            f"items={self.num_items}, "
            f"built=False)"
        )
