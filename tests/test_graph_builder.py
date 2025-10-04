"""
Unit tests for InteractionGraphBuilder.

Tests cover:
- Graph construction from sample data
- Edge index shape and bidirectionality
- Node feature generation
- Graph statistics computation
- Error handling for invalid inputs
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.graph.graph_builder import InteractionGraphBuilder


@pytest.fixture
def sample_interactions():
    """Create sample interaction data for testing."""
    data = {
        "user_idx": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "item_idx": [0, 1, 1, 2, 0, 3, 2, 4, 3, 4],
        "user_id": ["u0", "u0", "u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
        "item_id": ["i0", "i1", "i1", "i2", "i0", "i3", "i2", "i4", "i3", "i4"],
        "title": [f"Product {i}" for i in range(10)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def graph_builder(sample_interactions):
    """Create a GraphBuilder instance with sample data."""
    return InteractionGraphBuilder(
        interactions_df=sample_interactions,
        num_users=5,
        num_items=5,
    )


class TestGraphConstruction:
    """Tests for basic graph construction."""

    def test_graph_builds_successfully(self, graph_builder):
        """Test that a graph can be built from valid interactions."""
        graph = graph_builder.build_graph(embedding_dim=64)

        assert graph is not None
        assert graph.x is not None
        assert graph.edge_index is not None
        assert graph.num_nodes == 10  # 5 users + 5 items

    def test_node_count_matches(self, graph_builder):
        """Test that total nodes = num_users + num_items."""
        graph = graph_builder.build_graph(embedding_dim=64)
        assert graph.num_nodes == graph_builder.num_users + graph_builder.num_items

    def test_graph_data_stored(self, graph_builder):
        """Test that graph_data attribute is set after build."""
        assert graph_builder.graph_data is None
        graph_builder.build_graph(embedding_dim=64)
        assert graph_builder.graph_data is not None


class TestEdgeIndex:
    """Tests for edge index construction."""

    def test_edge_index_shape(self, graph_builder):
        """Test edge index has correct shape (2, num_edges)."""
        graph = graph_builder.build_graph(embedding_dim=64)
        edge_index = graph.edge_index

        assert edge_index.dim() == 2
        assert edge_index.shape[0] == 2  # Source and destination rows

    def test_bidirectional_edges(self, graph_builder):
        """Test that reverse edges are included for bidirectional message passing."""
        graph = graph_builder.build_graph(embedding_dim=64)
        edge_index = graph.edge_index

        # For each (u, i) edge, there should be a (i, u) reverse edge
        edge_set = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))

        # Check at least some reverse edges exist
        has_reverse = False
        for src, dst in edge_set:
            if (dst, src) in edge_set:
                has_reverse = True
                break

        assert has_reverse, "No reverse edges found — graph should be bidirectional"

    def test_edge_index_dtype(self, graph_builder):
        """Test edge index uses long integer type."""
        graph = graph_builder.build_graph(embedding_dim=64)
        assert graph.edge_index.dtype == torch.long

    def test_no_self_loops(self, graph_builder):
        """Test that there are no self-loops in the graph."""
        graph = graph_builder.build_graph(embedding_dim=64)
        edge_index = graph.edge_index

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            assert src != dst, f"Self-loop found at edge {i}: ({src}, {dst})"


class TestNodeFeatures:
    """Tests for node feature generation."""

    def test_node_features_shape(self, graph_builder):
        """Test node features have correct shape (num_nodes, embedding_dim)."""
        embedding_dim = 64
        graph = graph_builder.build_graph(embedding_dim=embedding_dim)

        assert graph.x.shape == (10, embedding_dim)

    def test_node_features_dtype(self, graph_builder):
        """Test node features are float tensors."""
        graph = graph_builder.build_graph(embedding_dim=64)
        assert graph.x.dtype == torch.float32

    def test_different_embedding_dims(self, graph_builder):
        """Test graph works with various embedding dimensions."""
        for dim in [32, 64, 128, 256]:
            graph = graph_builder.build_graph(embedding_dim=dim)
            assert graph.x.shape[1] == dim

    def test_node_type_labels(self, graph_builder):
        """Test that node type labels distinguish users from items."""
        graph = graph_builder.build_graph(embedding_dim=64)
        node_types = graph.node_type

        assert node_types.shape[0] == 10
        assert (node_types[:5] == 0).all()  # Users are type 0
        assert (node_types[5:] == 1).all()  # Items are type 1


class TestGraphStatistics:
    """Tests for graph statistics computation."""

    def test_statistics_keys(self, graph_builder):
        """Test that statistics dict has all expected keys."""
        graph_builder.build_graph(embedding_dim=64)
        stats = graph_builder.get_graph_statistics()

        expected_keys = [
            "num_nodes", "num_user_nodes", "num_item_nodes",
            "num_edges", "num_interactions", "avg_degree",
            "user_avg_degree", "item_avg_degree",
            "user_max_degree", "item_max_degree", "density",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_node_counts_in_stats(self, graph_builder):
        """Test that node counts in statistics are correct."""
        graph_builder.build_graph(embedding_dim=64)
        stats = graph_builder.get_graph_statistics()

        assert stats["num_user_nodes"] == 5
        assert stats["num_item_nodes"] == 5
        assert stats["num_nodes"] == 10

    def test_statistics_before_build_raises(self, graph_builder):
        """Test that getting stats before building raises error."""
        with pytest.raises(RuntimeError, match="not built"):
            graph_builder.get_graph_statistics()


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_missing_column_raises(self, sample_interactions):
        """Test that missing column in DataFrame raises ValueError."""
        bad_df = sample_interactions.drop(columns=["user_idx"])
        with pytest.raises(ValueError, match="not found"):
            InteractionGraphBuilder(
                interactions_df=bad_df,
                num_users=5,
                num_items=5,
            )

    def test_zero_users_raises(self, sample_interactions):
        """Test that zero users raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            InteractionGraphBuilder(
                interactions_df=sample_interactions,
                num_users=0,
                num_items=5,
            )

    def test_repr_before_build(self, graph_builder):
        """Test string representation before building."""
        repr_str = repr(graph_builder)
        assert "built=False" in repr_str

    def test_repr_after_build(self, graph_builder):
        """Test string representation after building."""
        graph_builder.build_graph(embedding_dim=64)
        repr_str = repr(graph_builder)
        assert "nodes=" in repr_str
        assert "edges=" in repr_str


class TestTrainTestEdgeSplit:
    """Tests for edge splitting functionality."""

    def test_split_produces_three_tensors(self, graph_builder):
        """Test that edge split returns train, pos, neg tensors."""
        graph_builder.build_graph(embedding_dim=64)
        train_ei, test_pos, test_neg = graph_builder.get_train_test_edges(test_ratio=0.2)

        assert train_ei.dim() == 2 and train_ei.shape[0] == 2
        assert test_pos.dim() == 2 and test_pos.shape[0] == 2
        assert test_neg.dim() == 2 and test_neg.shape[0] == 2

    def test_split_sizes(self, graph_builder):
        """Test that split sizes are reasonable."""
        graph_builder.build_graph(embedding_dim=64)
        train_ei, test_pos, test_neg = graph_builder.get_train_test_edges(test_ratio=0.2)

        # Test set should be smaller than train set
        assert test_pos.shape[1] > 0
        assert test_neg.shape[1] > 0
        assert train_ei.shape[1] > test_pos.shape[1]

    def test_split_before_build_raises(self, graph_builder):
        """Test that splitting before building raises error."""
        with pytest.raises(RuntimeError, match="not built"):
            graph_builder.get_train_test_edges()
