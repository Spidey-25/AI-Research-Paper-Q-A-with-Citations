"""Shared pytest fixtures.

Tests run entirely on the offline providers, so the suite needs no Ollama
server, no API key and no network access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Force the deterministic providers before rag.config is imported anywhere.
os.environ.setdefault("RAG_EMBEDDING_PROVIDER", "offline")
os.environ.setdefault("RAG_LLM_PROVIDER", "offline")

from langchain_core.documents import Document  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def sample_docs() -> list[Document]:
    """A miniature two-paper corpus with fully populated citation metadata."""
    return [
        Document(
            page_content=(
                "The encoder is composed of a stack of N=6 identical layers. Each layer has "
                "two sub-layers, a multi-head self-attention mechanism and a position-wise "
                "fully connected feed-forward network."
            ),
            metadata={
                "source": "1706.03762.pdf",
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762",
                "page_number": 3,
                "url": "https://arxiv.org/abs/1706.03762",
            },
        ),
        Document(
            page_content=(
                "We mask 15% of all WordPiece tokens in each sequence at random and predict "
                "the masked tokens. BERT-BASE has 110M parameters and BERT-LARGE has 340M "
                "parameters."
            ),
            metadata={
                "source": "1810.04805.pdf",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "arxiv_id": "1810.04805",
                "page_number": 4,
                "url": "https://arxiv.org/abs/1810.04805",
            },
        ),
    ]


@pytest.fixture
def temp_store(tmp_path, sample_docs):
    """A real Chroma collection over ``sample_docs``, built with offline embeddings."""
    from langchain_chroma import Chroma

    from rag import config
    from rag.providers import HashingEmbeddings

    return Chroma.from_documents(
        documents=sample_docs,
        embedding=HashingEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
