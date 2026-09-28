"""Central configuration for the RAG pipeline.

Every tunable is read from the environment with a sensible default, so the
package behaves identically whether it is driven from the CLI, from the test
suite, or from the evaluation harness.

Paths are resolved relative to this file rather than the current working
directory, so `python -m rag.query ...` works from anywhere on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root: src/rag/config.py -> src/rag -> src -> <repo root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Load .env if one exists, so ANTHROPIC_API_KEY and friends can live in a file
# that is gitignored. Optional: the offline providers need no secrets at all,
# so a missing python-dotenv must not be fatal.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

# --------------------------------------------------------------------------
# Paths (all __file__-relative, never CWD-relative)
# --------------------------------------------------------------------------

CORPUS_DIR: Path = Path(os.getenv("RAG_CORPUS_DIR", REPO_ROOT / "corpus"))
PDF_DIR: Path = Path(os.getenv("RAG_PDF_DIR", CORPUS_DIR / "pdfs"))
MANIFEST_PATH: Path = Path(os.getenv("RAG_MANIFEST", CORPUS_DIR / "manifest.json"))
CHROMA_DIR: Path = Path(os.getenv("RAG_CHROMA_DIR", REPO_ROOT / "chroma_db"))

#: Chroma collection name. Bumping this is the cheapest way to invalidate an
#: index built with a different embedding model.
COLLECTION_NAME: str = os.getenv("RAG_COLLECTION", "arxiv_papers")

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

#: Similarity floor in [0, 1]. Chunks scoring below this are discarded before
#: they ever reach the prompt; if nothing survives, the pipeline refuses
#: instead of asking the model to answer from irrelevant context. This is the
#: first of two anti-hallucination gates (the second is the prompt itself).
MIN_RELEVANCE: float = float(os.getenv("RAG_MIN_RELEVANCE", "0.0"))

# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

#: "ollama" (real, needs a local Ollama server) or "offline" (deterministic,
#: zero-dependency, used by CI, the test suite and the eval harness).
EMBEDDING_PROVIDER: str = os.getenv("RAG_EMBEDDING_PROVIDER", "offline").lower()

#: "anthropic" (real, needs ANTHROPIC_API_KEY) or "offline" (deterministic
#: extractive reader).
LLM_PROVIDER: str = os.getenv("RAG_LLM_PROVIDER", "offline").lower()

OLLAMA_EMBEDDING_MODEL: str = os.getenv("RAG_OLLAMA_MODEL", "embeddinggemma")
OLLAMA_BATCH_SIZE: int = int(os.getenv("RAG_OLLAMA_BATCH_SIZE", "50"))
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

#: Anthropic model id. Deliberately configurable rather than hardcoded:
#: Anthropic's model identifiers are versioned and are superseded over time, so
#: pinning one in source guarantees the repo breaks eventually. The default
#: below was checked against Anthropic's published model list at the time of
#: writing; verify the current id at
#: https://platform.claude.com/docs/en/about-claude/models/overview
#: (An earlier draft of this project hardcoded "claude-sonnet-4-6", which was
#: never a valid model id - hence this indirection.)
ANTHROPIC_MODEL: str = os.getenv("RAG_LLM_MODEL", "claude-sonnet-5")
LLM_MAX_TOKENS: int = int(os.getenv("RAG_LLM_MAX_TOKENS", "1024"))

#: Offline embedding dimensionality. 1024 hashed buckets is ample for a
#: corpus of this size and keeps the index small.
OFFLINE_EMBEDDING_DIM: int = int(os.getenv("RAG_OFFLINE_DIM", "1024"))

#: How much of a question's (IDF-weighted) content must appear in a passage
#: before the offline reader will answer from it rather than refuse.
#: Deliberately set on the conservative side: a false refusal is a visible
#: nuisance, a confident wrong answer is a silent failure. See eval/RESULTS.md
#: for the measured trade-off - this is the one hyperparameter tuned against
#: the evaluation set, and it is disclosed there.
OFFLINE_COVERAGE_THRESHOLD: float = float(os.getenv("RAG_OFFLINE_COVERAGE", "0.40"))

# --------------------------------------------------------------------------
# Answer contract
# --------------------------------------------------------------------------

#: The exact string the system must emit when the answer is not supported by
#: retrieved context. The evaluation harness scores refusal accuracy against
#: this constant, so it lives in exactly one place.
REFUSAL_TEXT: str = os.getenv(
    "RAG_REFUSAL_TEXT",
    "I could not find an answer to that in the indexed documents.",
)

SYSTEM_PROMPT: str = """You are a careful research assistant answering questions about a \
corpus of machine-learning papers.

Rules you must follow:
1. Answer using ONLY the numbered context passages below. Never use outside knowledge.
2. Cite every claim inline with the bracketed number of the passage it came from, \
for example: "The encoder is a stack of six identical layers [2]."
3. If the context passages do not contain the answer, reply with exactly this \
sentence and nothing else: "{refusal}"
4. Be concise. Prefer two or three sentences over a paragraph.

Context passages:
{context}"""


def describe() -> dict:
    """Return the active configuration - handy for `--verbose` and for RESULTS.md."""
    return {
        "embedding_provider": EMBEDDING_PROVIDER,
        "llm_provider": LLM_PROVIDER,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "min_relevance": MIN_RELEVANCE,
        "collection": COLLECTION_NAME,
        "chroma_dir": str(CHROMA_DIR),
    }
