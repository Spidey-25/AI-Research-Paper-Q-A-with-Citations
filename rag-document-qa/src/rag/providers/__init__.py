"""Pluggable backends for embeddings and answer generation.

Both the real (Ollama / Anthropic) and the offline deterministic
implementations satisfy the same interfaces, so the rest of the package never
branches on which one is active.
"""

from .embeddings import (
    BatchedOllamaEmbeddings,
    EmbeddingProviderError,
    HashingEmbeddings,
    get_embeddings,
)
from .llm import LLMProviderError, OfflineExtractiveLLM, get_llm

__all__ = [
    "BatchedOllamaEmbeddings",
    "HashingEmbeddings",
    "EmbeddingProviderError",
    "get_embeddings",
    "OfflineExtractiveLLM",
    "LLMProviderError",
    "get_llm",
]
