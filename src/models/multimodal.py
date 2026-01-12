"""
CLIP Vision Encoder for Multimodal Recommendations.

Extracts visual features from product images using OpenAI's CLIP model.
These features are fused with text (LLM) and graph (GNN) embeddings
for multimodal recommendation.

Architecture:
    [Product Image] → [CLIP ViT] → [Visual Embedding]
                                          ↓
    [GNN Embedding] + [LLM Embedding] + [Visual Embedding]
                                          ↓
                                  [Multimodal Fusion]

References:
    - Radford et al., "Learning Transferable Visual Models" (2021)
    - https://openai.com/research/clip
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class CLIPVisionEncoder:
    """
    Extract visual features from product images using CLIP.

    Supports lazy loading to conserve memory when vision features
    aren't needed. Can process single images, batches, and URLs.

    Attributes:
        model_name: CLIP model identifier.
        embedding_dim: Output embedding dimension.
        model: Loaded CLIP model (None until load_model called).
        processor: Image preprocessor.
        device: Computing device.

    Example:
        >>> encoder = CLIPVisionEncoder()
        >>> encoder.load_model()
        >>> embedding = encoder.encode_image("product.jpg")
        >>> print(embedding.shape)  # (512,)
    """

    KNOWN_DIMS = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-base-patch16": 512,
        "openai/clip-vit-large-patch14": 768,
        "openai/clip-vit-large-patch14-336": 768,
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize CLIP vision encoder.

        Args:
            model_name: HuggingFace model identifier.
            device: Computing device ('cuda', 'cpu', or auto-detect).
        """
        self.model_name = model_name
        self.embedding_dim = self.KNOWN_DIMS.get(model_name, 512)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.processor = None
        self._is_loaded = False

        logger.info(
            f"CLIPVisionEncoder initialized | "
            f"model={model_name}, dim={self.embedding_dim}"
        )

    def load_model(self) -> None:
        """
        Load CLIP model and processor from HuggingFace.

        Raises:
            ImportError: If transformers isn't installed.
        """
        if self._is_loaded:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers required. Install: pip install transformers"
            )

        logger.info(f"Loading CLIP model: {self.model_name}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True
        logger.info(f"CLIP model loaded on {self.device}")

    @torch.no_grad()
    def encode_image(
        self,
        image: Any,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a single image to a feature vector.

        Args:
            image: PIL Image, file path, or numpy array.
            normalize: Whether to L2-normalize the output.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from PIL import Image as PILImage

        # Handle different input types
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = PILImage.fromarray(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten()

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    @torch.no_grad()
    def encode_batch(
        self,
        images: List[Any],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a batch of images.

        Args:
            images: List of PIL Images, paths, or arrays.
            normalize: Whether to L2-normalize outputs.
            batch_size: Processing batch size.

        Returns:
            2D numpy array of shape (num_images, embedding_dim).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from PIL import Image as PILImage

        # Preprocess all images
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(PILImage.open(img).convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(PILImage.fromarray(img).convert("RGB"))
            else:
                pil_images.append(img)

        all_embeddings = []

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.get_image_features(**inputs)
            embeddings = outputs.cpu().numpy()

            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text descriptions using CLIP's text encoder.

        Useful for computing text-image similarity or as a fallback
        when product images are unavailable.

        Args:
            texts: List of text strings.
            normalize: Whether to L2-normalize outputs.

        Returns:
            2D numpy array of shape (num_texts, embedding_dim).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.get_text_features(**inputs)
        embeddings = outputs.cpu().numpy()

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        return embeddings

    def get_random_embedding(self) -> np.ndarray:
        """Generate a random embedding for testing without model loading."""
        rng = np.random.RandomState(42)
        emb = rng.randn(self.embedding_dim).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not_loaded"
        return (
            f"CLIPVisionEncoder("
            f"model='{self.model_name}', "
            f"dim={self.embedding_dim}, "
            f"status={status})"
        )


class MultimodalFusionLayer(nn.Module):
    """
    Fuse GNN, LLM, and Vision embeddings for trimodal recommendations.

    Uses learned attention weights to combine three embedding modalities:
    - GNN: Graph structure (collaborative filtering signal)
    - LLM: Text semantics (product descriptions)
    - Vision: Visual features (product images)

    Architecture:
        [GNN_emb]    → [proj_gnn]    ─┐
        [LLM_emb]    → [proj_llm]    ─┤→ [Attention] → [Weighted Sum] → [MLP] → output
        [Vision_emb] → [proj_vision] ─┘

    Attributes:
        gnn_dim: GNN embedding dimension.
        llm_dim: LLM embedding dimension.
        vision_dim: Vision embedding dimension.
        fusion_dim: Projected fusion dimension.
        output_dim: Final output dimension.
    """

    def __init__(
        self,
        gnn_dim: int = 128,
        llm_dim: int = 768,
        vision_dim: int = 512,
        fusion_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.gnn_dim = gnn_dim
        self.llm_dim = llm_dim
        self.vision_dim = vision_dim
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim

        # Projection layers to common dimension
        self.proj_gnn = nn.Sequential(
            nn.Linear(gnn_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.proj_llm = nn.Sequential(
            nn.Linear(llm_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.proj_vision = nn.Sequential(
            nn.Linear(vision_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

        # Attention mechanism for modality weighting
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 3, 3),
            nn.Softmax(dim=-1),
        )

        # Output MLP
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, output_dim),
        )

    def forward(
        self,
        gnn_emb: torch.Tensor,
        llm_emb: torch.Tensor,
        vision_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse three modality embeddings.

        Args:
            gnn_emb: (batch, gnn_dim) graph embeddings.
            llm_emb: (batch, llm_dim) text embeddings.
            vision_emb: (batch, vision_dim) visual embeddings.

        Returns:
            (batch, output_dim) fused embeddings.
        """
        # Project to common dimension
        gnn_proj = self.proj_gnn(gnn_emb)       # (batch, fusion_dim)
        llm_proj = self.proj_llm(llm_emb)       # (batch, fusion_dim)
        vis_proj = self.proj_vision(vision_emb)  # (batch, fusion_dim)

        # Compute attention weights
        concat = torch.cat([gnn_proj, llm_proj, vis_proj], dim=-1)
        weights = self.attention(concat)  # (batch, 3)

        # Weighted sum
        w_gnn = weights[:, 0:1]    # (batch, 1)
        w_llm = weights[:, 1:2]
        w_vis = weights[:, 2:3]

        fused = w_gnn * gnn_proj + w_llm * llm_proj + w_vis * vis_proj

        return self.output_head(fused)

    def forward_with_weights(
        self,
        gnn_emb: torch.Tensor,
        llm_emb: torch.Tensor,
        vision_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and attention weights.

        Returns:
            Tuple of (output, attention_weights).
            attention_weights shape: (batch, 3) for [gnn, llm, vision].
        """
        gnn_proj = self.proj_gnn(gnn_emb)
        llm_proj = self.proj_llm(llm_emb)
        vis_proj = self.proj_vision(vision_emb)

        concat = torch.cat([gnn_proj, llm_proj, vis_proj], dim=-1)
        weights = self.attention(concat)

        w_gnn = weights[:, 0:1]
        w_llm = weights[:, 1:2]
        w_vis = weights[:, 2:3]

        fused = w_gnn * gnn_proj + w_llm * llm_proj + w_vis * vis_proj
        output = self.output_head(fused)

        return output, weights

    def get_modality_importance(
        self,
        gnn_emb: torch.Tensor,
        llm_emb: torch.Tensor,
        vision_emb: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute average modality importance weights.

        Returns:
            Dict with mean weights for gnn, llm, vision.
        """
        _, weights = self.forward_with_weights(gnn_emb, llm_emb, vision_emb)
        w_mean = weights.mean(dim=0)

        return {
            "gnn_weight": float(w_mean[0]),
            "llm_weight": float(w_mean[1]),
            "vision_weight": float(w_mean[2]),
        }

    def __repr__(self) -> str:
        return (
            f"MultimodalFusionLayer("
            f"gnn={self.gnn_dim}, llm={self.llm_dim}, "
            f"vision={self.vision_dim}, out={self.output_dim})"
        )
