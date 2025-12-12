"""
Unit tests for multimodal components.

Tests cover:
- CLIPVisionEncoder random embedding generation
- MultimodalFusionLayer forward pass
- Attention weight computation
- Modality importance analysis
- Gradient flow through all three branches
"""

import pytest
import numpy as np
import torch

from src.models.multimodal import CLIPVisionEncoder, MultimodalFusionLayer


class TestCLIPVisionEncoder:
    """Tests for CLIP vision encoder (without loading actual model)."""

    def test_init(self):
        """Test encoder initialization."""
        encoder = CLIPVisionEncoder()
        assert encoder.embedding_dim == 512
        assert not encoder._is_loaded

    def test_known_dims(self):
        """Test known model dimensions."""
        enc_base = CLIPVisionEncoder("openai/clip-vit-base-patch32")
        assert enc_base.embedding_dim == 512

        enc_large = CLIPVisionEncoder("openai/clip-vit-large-patch14")
        assert enc_large.embedding_dim == 768

    def test_random_embedding(self):
        """Test random embedding generation for testing."""
        encoder = CLIPVisionEncoder()
        emb = encoder.get_random_embedding()

        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        # Should be L2-normalized
        assert abs(np.linalg.norm(emb) - 1.0) < 0.01

    def test_encode_without_load_raises(self):
        """Test that encoding without loading raises error."""
        encoder = CLIPVisionEncoder()
        with pytest.raises(RuntimeError, match="not loaded"):
            encoder.encode_image(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_repr(self):
        """Test string representation."""
        encoder = CLIPVisionEncoder()
        assert "CLIPVisionEncoder" in repr(encoder)
        assert "not_loaded" in repr(encoder)


class TestMultimodalFusion:
    """Tests for MultimodalFusionLayer."""

    @pytest.fixture
    def fusion_layer(self):
        return MultimodalFusionLayer(
            gnn_dim=128, llm_dim=768, vision_dim=512,
            fusion_dim=256, output_dim=128,
        )

    @pytest.fixture
    def sample_embeddings(self):
        batch = 8
        return (
            torch.randn(batch, 128),   # GNN
            torch.randn(batch, 768),   # LLM
            torch.randn(batch, 512),   # Vision
        )

    def test_forward_shape(self, fusion_layer, sample_embeddings):
        """Test output shape is (batch, output_dim)."""
        gnn, llm, vis = sample_embeddings
        output = fusion_layer(gnn, llm, vis)
        assert output.shape == (8, 128)

    def test_no_nan_output(self, fusion_layer, sample_embeddings):
        """Test no NaN in outputs."""
        output = fusion_layer(*sample_embeddings)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_weights(self, fusion_layer, sample_embeddings):
        """Test forward with attention weights."""
        output, weights = fusion_layer.forward_with_weights(*sample_embeddings)
        assert output.shape == (8, 128)
        assert weights.shape == (8, 3)

        # Weights should sum to 1 (softmax)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(8), atol=1e-5)

    def test_attention_weights_positive(self, fusion_layer, sample_embeddings):
        """Test that all attention weights are positive."""
        _, weights = fusion_layer.forward_with_weights(*sample_embeddings)
        assert (weights >= 0).all()

    def test_modality_importance(self, fusion_layer, sample_embeddings):
        """Test modality importance analysis."""
        importance = fusion_layer.get_modality_importance(*sample_embeddings)

        assert "gnn_weight" in importance
        assert "llm_weight" in importance
        assert "vision_weight" in importance
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_gradient_flow_all_branches(self, fusion_layer, sample_embeddings):
        """Test gradient flows through all three modalities."""
        gnn, llm, vis = sample_embeddings
        gnn.requires_grad_(True)
        llm.requires_grad_(True)
        vis.requires_grad_(True)

        output = fusion_layer(gnn, llm, vis)
        loss = output.sum()
        loss.backward()

        assert gnn.grad is not None and gnn.grad.abs().sum() > 0
        assert llm.grad is not None and llm.grad.abs().sum() > 0
        assert vis.grad is not None and vis.grad.abs().sum() > 0

    def test_eval_deterministic(self, fusion_layer, sample_embeddings):
        """Test eval mode is deterministic."""
        fusion_layer.eval()
        with torch.no_grad():
            out1 = fusion_layer(*sample_embeddings)
            out2 = fusion_layer(*sample_embeddings)
        assert torch.allclose(out1, out2)

    def test_different_batch_sizes(self, fusion_layer):
        """Test with various batch sizes."""
        for bs in [1, 4, 16, 64]:
            gnn = torch.randn(bs, 128)
            llm = torch.randn(bs, 768)
            vis = torch.randn(bs, 512)
            out = fusion_layer(gnn, llm, vis)
            assert out.shape == (bs, 128)

    def test_repr(self, fusion_layer):
        """Test string representation."""
        assert "MultimodalFusionLayer" in repr(fusion_layer)
        assert "vision=512" in repr(fusion_layer)
