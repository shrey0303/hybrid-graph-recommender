"""Model architectures for hybrid recommendation."""

from src.models.hybrid_model import HybridGNN_LLM
from src.models.llm_wrapper import LLMEmbeddingExtractor

__all__ = ["HybridGNN_LLM", "LLMEmbeddingExtractor"]
