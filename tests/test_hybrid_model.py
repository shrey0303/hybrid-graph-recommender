"""
Unit tests for HybridGNN_LLM fusion model.

Tests cover:
- Fusion forward pass
- Gated fusion mechanism (α values)
- Output dimensionality
- Gradient flow through both branches
- Interpretability methods
"""

import pytest
import torch

from src.models.hybrid_model import HybridGNN_LLM, GatedFusionLayer


@pytest.fixture
def hybrid_model():
    """Create a hybrid model for testing."""
    return HybridGNN_LLM(
        gnn_dim=128,
        llm_dim=768,
        fusion_dim=256,
        output_dim=128,
        num_mlp_layers=2,
        dropout=0.1,
    )


@pytest.fixture
def sample_embeddings():
    """Create sample GNN and LLM embeddings."""
    batch_size = 16
    gnn_emb = torch.randn(batch_size, 128)
    llm_emb = torch.randn(batch_size, 768)
    return gnn_emb, llm_emb


class TestGatedFusion:
    """Tests for gated fusion mechanism."""

    def test_gate_values_range(self):
        """Test that gate values are between 0 and 1 (sigmoid output)."""
        fusion = GatedFusionLayer(
            input_dim_a=64, input_dim_b=128, projection_dim=64
        )

        a = torch.randn(8, 64)
        b = torch.randn(8, 128)

        _, gate_values = fusion(a, b)

        assert (gate_values >= 0).all(), "Gate values should be >= 0"
        assert (gate_values <= 1).all(), "Gate values should be <= 1"

    def test_fusion_output_shape(self):
        """Test fusion output has correct shape."""
        fusion = GatedFusionLayer(
            input_dim_a=64, input_dim_b=128, projection_dim=64
        )

        a = torch.randn(8, 64)
        b = torch.randn(8, 128)

        fused, gate = fusion(a, b)

        assert fused.shape == (8, 64)
        assert gate.shape == (8, 1)


class TestHybridModel:
    """Tests for HybridGNN_LLM model."""

    def test_forward_pass(self, hybrid_model, sample_embeddings):
        """Test forward pass produces output of correct shape."""
        gnn_emb, llm_emb = sample_embeddings
        output = hybrid_model(gnn_emb, llm_emb)

        assert output.shape == (16, 128)  # (batch_size, output_dim)

    def test_forward_with_gate(self, hybrid_model, sample_embeddings):
        """Test forward_with_gate returns both output and gate values."""
        gnn_emb, llm_emb = sample_embeddings
        output, gate_values = hybrid_model.forward_with_gate(gnn_emb, llm_emb)

        assert output.shape == (16, 128)
        assert gate_values.shape == (16, 1)
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()

    def test_no_nan_output(self, hybrid_model, sample_embeddings):
        """Test that output contains no NaN values."""
        gnn_emb, llm_emb = sample_embeddings
        output = hybrid_model(gnn_emb, llm_emb)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, hybrid_model, sample_embeddings):
        """Test gradient flow through both GNN and LLM branches."""
        gnn_emb, llm_emb = sample_embeddings
        gnn_emb.requires_grad_(True)
        llm_emb.requires_grad_(True)

        output = hybrid_model(gnn_emb, llm_emb)
        loss = output.sum()
        loss.backward()

        assert gnn_emb.grad is not None, "No gradient for GNN embeddings"
        assert llm_emb.grad is not None, "No gradient for LLM embeddings"
        assert gnn_emb.grad.abs().sum() > 0, "Zero gradients for GNN"
        assert llm_emb.grad.abs().sum() > 0, "Zero gradients for LLM"

    def test_different_batch_sizes(self, hybrid_model):
        """Test model works with various batch sizes."""
        for bs in [1, 4, 32, 64]:
            gnn_emb = torch.randn(bs, 128)
            llm_emb = torch.randn(bs, 768)
            output = hybrid_model(gnn_emb, llm_emb)
            assert output.shape == (bs, 128)

    def test_eval_mode_deterministic(self, hybrid_model, sample_embeddings):
        """Test that eval mode produces deterministic outputs."""
        gnn_emb, llm_emb = sample_embeddings

        hybrid_model.eval()
        with torch.no_grad():
            out1 = hybrid_model(gnn_emb, llm_emb)
            out2 = hybrid_model(gnn_emb, llm_emb)

        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


class TestFusionWeights:
    """Tests for fusion weight analysis."""

    def test_get_fusion_weights(self, hybrid_model, sample_embeddings):
        """Test fusion weight analysis method."""
        gnn_emb, llm_emb = sample_embeddings
        weights = hybrid_model.get_fusion_weights(gnn_emb, llm_emb)

        assert "gnn_weight_mean" in weights
        assert "llm_weight_mean" in weights
        assert 0 <= weights["gnn_weight_mean"] <= 1
        assert 0 <= weights["llm_weight_mean"] <= 1

    def test_weights_sum_to_one(self, hybrid_model, sample_embeddings):
        """Test that GNN and LLM weights approximately sum to 1."""
        gnn_emb, llm_emb = sample_embeddings
        weights = hybrid_model.get_fusion_weights(gnn_emb, llm_emb)

        total = weights["gnn_weight_mean"] + weights["llm_weight_mean"]
        assert abs(total - 1.0) < 0.01, f"Weights should sum to ~1, got {total}"


class TestModelConfig:
    """Tests for model configuration."""

    def test_various_fusion_dims(self):
        """Test model works with different fusion dimensions."""
        for fusion_dim in [64, 128, 256, 512]:
            model = HybridGNN_LLM(
                gnn_dim=128, llm_dim=768,
                fusion_dim=fusion_dim, output_dim=64,
            )
            gnn = torch.randn(4, 128)
            llm = torch.randn(4, 768)
            out = model(gnn, llm)
            assert out.shape == (4, 64)

    def test_repr(self, hybrid_model):
        """Test string representation."""
        repr_str = repr(hybrid_model)
        assert "HybridGNN_LLM" in repr_str
        assert "gnn=128d" in repr_str
        assert "llm=768d" in repr_str
