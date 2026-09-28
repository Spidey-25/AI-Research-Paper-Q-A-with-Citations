"""Embedding providers.

Two implementations sit behind one factory:

``ollama``
    :class:`BatchedOllamaEmbeddings` - the production path. Talks to a local
    Ollama server running ``embeddinggemma``. Chunks the request stream into
    small batches, because Ollama rejects oversized embedding requests and the
    naive one-shot call fails on any realistic corpus.

``offline``
    :class:`HashingEmbeddings` - a deterministic, dependency-free bag-of-words
    embedding. No server, no API key, no model download. Identical vectors on
    every machine and every run, which is what makes the test suite and the
    evaluation harness reproducible in CI.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings

from .. import config


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding backend cannot be reached or configured."""


class BatchedOllamaEmbeddings(Embeddings):
    """Sends texts to Ollama in small batches to avoid request size limits.

    Kept from the original prototype - it solves a real problem - but now it
    lives behind the provider interface and reports failures usefully instead
    of surfacing a raw connection error from deep inside LangChain.
    """

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:  # pragma: no cover - import guard
            raise EmbeddingProviderError(
                "langchain-ollama is not installed. Install requirements.txt, or set "
                "RAG_EMBEDDING_PROVIDER=offline to run without Ollama."
            ) from exc

        # Explicit None checks: `batch_size or default` would silently swallow
        # an invalid 0 and fall back to the default instead of rejecting it.
        self.model = model if model is not None else config.OLLAMA_EMBEDDING_MODEL
        self.batch_size = batch_size if batch_size is not None else config.OLLAMA_BATCH_SIZE
        self.base_url = base_url if base_url is not None else config.OLLAMA_BASE_URL
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self._emb = OllamaEmbeddings(model=self.model, base_url=self.base_url)

    def _fail(self, exc: Exception) -> EmbeddingProviderError:
        return EmbeddingProviderError(
            f"Could not embed with Ollama model '{self.model}' at {self.base_url}: {exc}\n"
            "Is the Ollama server running? Try:\n"
            f"  ollama serve\n  ollama pull {self.model}\n"
            "Or set RAG_EMBEDDING_PROVIDER=offline to run without Ollama."
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                results.extend(self._emb.embed_documents(batch))
            except Exception as exc:
                raise self._fail(exc) from exc
        return results

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._emb.embed_query(text)
        except Exception as exc:
            raise self._fail(exc) from exc


_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Very common words carry no retrieval signal in a bag-of-words model.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    to was were what when where which who why with does do did this these those there
    can could should would will shall may might must about into over under between
    """.split()
)


def _stem(token: str) -> str:
    """A deliberately conservative suffix stripper.

    Without it, a purely lexical retriever treats "masking" and "masked" as
    unrelated words - which cost this project two false refusals on the
    evaluation set before it was added. It is not a Porter stemmer; it only
    removes the handful of inflections that actually matter here, because
    aggressive stemming collapses distinct technical terms.
    """
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)]
        # Never strip down to a fragment: "does" must not become "do", and
        # short technical tokens must survive intact.
        if len(stem) < 3:
            continue
        return stem + "y" if suffix == "ies" else stem
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords removed, lightly stemmed."""
    return [
        _stem(t)
        for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


def _bucket(token: str, dim: int) -> int:
    """Stable hash bucket for a token.

    Uses blake2b rather than Python's built-in ``hash()``, which is salted per
    process - the built-in would silently produce a different index on every
    run and break persistence.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class HashingEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings with sublinear term weighting.

    Unigrams and adjacent bigrams are hashed into a fixed-width vector, weighted
    ``1 + log(tf)``, then L2-normalised so that inner product equals cosine
    similarity. It is not semantic - it will not match paraphrases the way a
    trained embedding model does - but it is a legitimate lexical retriever, it
    needs nothing but the standard library, and it is byte-for-byte reproducible.
    """

    def __init__(self, dim: int | None = None, use_bigrams: bool = True) -> None:
        self.dim = dim or config.OFFLINE_EMBEDDING_DIM
        if self.dim < 16:
            raise ValueError("embedding dim must be >= 16")
        self.use_bigrams = use_bigrams

    def _vector(self, text: str) -> List[float]:
        tokens = tokenize(text)
        if self.use_bigrams:
            tokens = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

        counts: dict[int, float] = {}
        for token in tokens:
            idx = _bucket(token, self.dim)
            counts[idx] = counts.get(idx, 0.0) + 1.0

        vec = [0.0] * self.dim
        for idx, tf in counts.items():
            vec[idx] = 1.0 + math.log(tf)

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # An empty or all-stopword chunk. Return a zero vector rather than
            # dividing by zero; Chroma handles it and it will simply never rank.
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Return the configured embedding backend.

    Selected by ``RAG_EMBEDDING_PROVIDER`` (``ollama`` | ``offline``).
    """
    name = (provider or config.EMBEDDING_PROVIDER).lower()
    if name == "ollama":
        return BatchedOllamaEmbeddings()
    if name == "offline":
        return HashingEmbeddings()
    raise EmbeddingProviderError(
        f"Unknown embedding provider {name!r}. Expected 'ollama' or 'offline'."
    )
