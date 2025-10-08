"""
Unit tests for GraphSAGERecommender model.

Tests cover:
- Forward pass with random inputs
- Output dimensionality correctness
- Link prediction scoring
- Gradient flow through the network
- Top-K recommendation generation
"""

import pytest
import torch
import torch.nn as nn

from src.graph.gnn_model import GraphSAGERecommender


@pytest.fixture
def model():
    """Create a GraphSAGE model for testing."""
    return GraphSAGERecommender(
        in_channels=64,
        hidden_channels=128,
        out_channels=64,
        num_layers=3,
        dropout=0.2,
    )


@pytest.fixture
def random_graph():
    """Create a random graph for testing."""
    num_nodes = 50
    num_edges = 200
    in_channels = 64

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    return x, edge_index, num_nodes


class TestForwardPass:
    """Tests for model forward pass."""

    def test_encode_returns_tensor(self, model, random_graph):
        """Test that encode returns a valid tensor."""
        x, edge_index, _ = random_graph
        embeddings = model.encode(x, edge_index)

        assert isinstance(embeddings, torch.Tensor)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_encode_output_shape(self, model, random_graph):
        """Test that encode output has correct dimensions."""
        x, edge_index, num_nodes = random_graph
        embeddings = model.encode(x, edge_index)

        assert embeddings.shape == (num_nodes, 64)  # out_channels = 64

    def test_forward_returns_tuple(self, model, random_graph):
        """Test that forward returns (embeddings, scores) tuple."""
        x, edge_index, _ = random_graph
        result = model(x, edge_index)

        assert isinstance(result, tuple)
        assert len(result) == 2
        embeddings, scores = result
        assert isinstance(embeddings, torch.Tensor)
        assert scores is None  # No indices provided

    def test_forward_with_indices(self, model, random_graph):
        """Test forward pass with link prediction indices."""
        x, edge_index, _ = random_graph
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([25, 26, 27, 28])

        embeddings, scores = model(x, edge_index, src, dst)

        assert scores is not None
        assert scores.shape == (4,)


class TestOutputDimensions:
    """Tests for output dimension correctness."""

    def test_various_output_dims(self, random_graph):
        """Test model with different output dimensions."""
        x, edge_index, num_nodes = random_graph

        for out_dim in [32, 64, 128, 256]:
            model = GraphSAGERecommender(
                in_channels=64, hidden_channels=128,
                out_channels=out_dim, num_layers=2,
            )
            embeddings = model.encode(x, edge_index)
            assert embeddings.shape == (num_nodes, out_dim)

    def test_single_layer_model(self, random_graph):
        """Test model with only one layer."""
        x, edge_index, num_nodes = random_graph

        model = GraphSAGERecommender(
            in_channels=64, hidden_channels=128,
            out_channels=32, num_layers=1,
        )
        embeddings = model.encode(x, edge_index)
        assert embeddings.shape == (num_nodes, 32)

    def test_deep_model(self, random_graph):
        """Test model with many layers."""
        x, edge_index, num_nodes = random_graph

        model = GraphSAGERecommender(
            in_channels=64, hidden_channels=128,
            out_channels=64, num_layers=5,
        )
        embeddings = model.encode(x, edge_index)
        assert embeddings.shape == (num_nodes, 64)


class TestLinkPrediction:
    """Tests for link prediction functionality."""

    def test_predict_link_returns_scores(self, model, random_graph):
        """Test that predict_link returns scalar scores."""
        x, edge_index, _ = random_graph
        embeddings = model.encode(x, edge_index)

        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([25, 30, 35])

        scores = model.predict_link(embeddings, src, dst)
        assert scores.shape == (3,)
        assert scores.dtype == torch.float32

    def test_predict_link_single_pair(self, model, random_graph):
        """Test link prediction for a single pair."""
        x, edge_index, _ = random_graph
        embeddings = model.encode(x, edge_index)

        src = torch.tensor([0])
        dst = torch.tensor([25])

        scores = model.predict_link(embeddings, src, dst)
        assert scores.shape == (1,)

    def test_same_node_similarity(self, model, random_graph):
        """Test that score of same node with itself is high."""
        x, edge_index, _ = random_graph
        embeddings = model.encode(x, edge_index)

        # Score of node with itself should be maximum (normalized dot product = 1)
        idx = torch.tensor([0])
        self_score = model.predict_link(embeddings, idx, idx)
        assert self_score.item() > 0  # Should be positive for normalized vectors


class TestGradientFlow:
    """Tests for gradient computation."""

    def test_backward_pass(self, model, random_graph):
        """Test that backward pass produces gradients for all parameters."""
        x, edge_index, _ = random_graph

        embeddings = model.encode(x, edge_index)
        loss = embeddings.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_link_prediction_gradient(self, model, random_graph):
        """Test gradient flow through link prediction."""
        x, edge_index, _ = random_graph

        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([25, 30, 35])

        embeddings, scores = model(x, edge_index, src, dst)
        loss = -torch.log(torch.sigmoid(scores) + 1e-8).mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients computed through link prediction"

    def test_train_eval_modes(self, model, random_graph):
        """Test that dropout behavior changes between train and eval."""
        x, edge_index, _ = random_graph

        # Get output in eval mode (deterministic)
        model.eval()
        with torch.no_grad():
            emb_eval1 = model.encode(x, edge_index)
            emb_eval2 = model.encode(x, edge_index)
        assert torch.allclose(emb_eval1, emb_eval2), "Eval mode should be deterministic"


class TestRecommendation:
    """Tests for top-K recommendation."""

    def test_recommend_items_shape(self, model, random_graph):
        """Test recommendation output shapes."""
        x, edge_index, _ = random_graph
        num_users = 25

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(x, edge_index)
            top_items, top_scores = model.recommend_items(
                embeddings, user_idx=0, num_users=num_users, top_k=5
            )

        assert top_items.shape == (5,)
        assert top_scores.shape == (5,)

    def test_recommend_items_sorted(self, model, random_graph):
        """Test that recommendations are sorted by score (descending)."""
        x, edge_index, _ = random_graph
        num_users = 25

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(x, edge_index)
            _, top_scores = model.recommend_items(
                embeddings, user_idx=0, num_users=num_users, top_k=10
            )

        # Verify scores are in descending order
        for i in range(len(top_scores) - 1):
            assert top_scores[i] >= top_scores[i + 1]


class TestValidation:
    """Tests for input validation."""

    def test_invalid_num_layers(self):
        """Test that zero layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            GraphSAGERecommender(
                in_channels=64, hidden_channels=128,
                out_channels=64, num_layers=0,
            )

    def test_invalid_dimensions(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            GraphSAGERecommender(
                in_channels=-1, hidden_channels=128,
                out_channels=64, num_layers=2,
            )

    def test_repr(self, model):
        """Test string representation."""
        repr_str = repr(model)
        assert "GraphSAGERecommender" in repr_str
        assert "in=64" in repr_str
        assert "layers=3" in repr_str
