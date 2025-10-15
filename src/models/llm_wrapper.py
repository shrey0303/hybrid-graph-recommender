"""
LLM Embedding Extractor Wrapper.

Provides a clean interface for extracting text embeddings from pre-trained
Large Language Models (TinyLlama, Mistral-7B, etc.) for use in the hybrid
GNN-LLM recommendation system.

Supports:
    - Mean-pooled last hidden state embeddings
    - Lazy model loading to conserve GPU memory
    - CPU/GPU automatic device placement
    - Batch processing for efficiency

Usage:
    >>> extractor = LLMEmbeddingExtractor("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    >>> embedding = extractor.get_text_embedding("Great product, highly recommend!")
    >>> print(embedding.shape)  # (1, 2048) for TinyLlama
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger


class LLMEmbeddingExtractor(nn.Module):
    """
    Extract text embeddings from pre-trained LLMs.

    Wraps a HuggingFace causal language model to extract mean-pooled
    hidden state representations suitable for use in downstream
    recommendation tasks.

    Attributes:
        model_name: HuggingFace model identifier.
        embedding_dim: Dimensionality of extracted embeddings.
        device: Computation device (CPU/GPU).
        is_loaded: Whether the model has been loaded into memory.

    Example:
        >>> extractor = LLMEmbeddingExtractor("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> extractor.load_model()
        >>> emb = extractor.get_text_embedding("Organic peanut butter, 16oz")
    """

    # Known embedding dimensions for common models
    KNOWN_DIMS: Dict[str, int] = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2048,
        "mistralai/Mistral-7B-v0.1": 4096,
        "mistralai/Mistral-7B-Instruct-v0.1": 4096,
        "meta-llama/Llama-2-7b-hf": 4096,
        "meta-llama/Llama-2-13b-hf": 5120,
    }

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
        max_length: int = 512,
        load_in_8bit: bool = False,
    ) -> None:
        """
        Initialize the LLM embedding extractor.

        Args:
            model_name: HuggingFace model name or path to local model.
            device: Device to load model on ('cuda', 'cpu', or None for auto).
            max_length: Maximum token sequence length.
            load_in_8bit: Whether to use 8-bit quantization (saves memory).
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.load_in_8bit = load_in_8bit

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Look up known embedding dimension
        self.embedding_dim = self.KNOWN_DIMS.get(model_name, 2048)

        # Lazy loading: model is loaded only when needed
        self._model = None
        self._tokenizer = None
        self.is_loaded = False

        logger.info(
            f"LLMEmbeddingExtractor initialized | "
            f"Model: {model_name} | Device: {self.device} | "
            f"Expected dim: {self.embedding_dim}"
        )

    def load_model(self) -> None:
        """
        Load the model and tokenizer into memory.

        This is separated from __init__ to allow lazy loading —
        the model is only loaded when embeddings are actually needed.

        Raises:
            ImportError: If transformers library is not installed.
            RuntimeError: If model loading fails.
        """
        if self.is_loaded:
            logger.info("Model already loaded, skipping.")
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading model {self.model_name}...")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Set pad token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            model_kwargs = {"trust_remote_code": True}

            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16

            self._model = AutoModel.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

            if not self.load_in_8bit:
                self._model = self._model.to(self.device)

            self._model.eval()
            self.is_loaded = True

            # Update embedding dim from actual model config
            if hasattr(self._model.config, "hidden_size"):
                self.embedding_dim = self._model.config.hidden_size

            logger.info(
                f"Model loaded successfully | "
                f"Embedding dim: {self.embedding_dim} | "
                f"Device: {self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    @torch.no_grad()
    def get_text_embedding(
        self,
        text: str,
    ) -> torch.Tensor:
        """
        Extract embedding from a single text input.

        Uses mean pooling over non-padding token positions in the
        last hidden state to produce a fixed-size representation.

        Args:
            text: Input text string.

        Returns:
            Embedding tensor of shape (1, embedding_dim).

        Raises:
            RuntimeError: If model hasn't been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or use "
                "get_random_embedding() for testing."
            )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self._model(**inputs)

        # Mean pooling over sequence length (excluding padding)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden_states = outputs.last_hidden_state
        masked_hidden = hidden_states * attention_mask
        embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

        return embedding.cpu()

    @torch.no_grad()
    def get_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Extract embeddings for a batch of texts.

        Processes texts in mini-batches for memory efficiency.

        Args:
            texts: List of input text strings.
            batch_size: Number of texts to process simultaneously.

        Returns:
            Embedding tensor of shape (num_texts, embedding_dim).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            outputs = self._model(**inputs)

            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden_states = outputs.last_hidden_state
            masked_hidden = hidden_states * attention_mask
            embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def get_random_embedding(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Generate random embeddings matching the model's output dimensions.

        Useful for testing the hybrid pipeline without loading the full
        LLM into memory.

        Args:
            batch_size: Number of random embeddings to generate.

        Returns:
            Random embedding tensor of shape (batch_size, embedding_dim).
        """
        embedding = torch.randn(batch_size, self.embedding_dim)
        return F.normalize(embedding, p=2, dim=-1)

    def forward(
        self,
        texts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Forward pass: extract embeddings from text input(s).

        Args:
            texts: Single text string or list of texts.

        Returns:
            Embedding tensor(s).
        """
        if isinstance(texts, str):
            return self.get_text_embedding(texts)
        return self.get_batch_embeddings(texts)

    def __repr__(self) -> str:
        return (
            f"LLMEmbeddingExtractor("
            f"model='{self.model_name}', "
            f"dim={self.embedding_dim}, "
            f"loaded={self.is_loaded})"
        )


# Import F for the random embedding normalization
import torch.nn.functional as F
