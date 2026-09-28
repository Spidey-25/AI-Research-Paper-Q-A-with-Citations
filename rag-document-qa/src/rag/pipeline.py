"""The retrieval-augmented answering pipeline.

Shared by the ``query`` CLI and the evaluation harness so that the thing being
evaluated is exactly the thing that ships.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .citations import AnswerBundle, build_blocks, dedupe_sources, extract_citation_numbers, format_context


class IndexNotFoundError(RuntimeError):
    """Raised when the vector store has not been built yet."""


def load_vectorstore(persist_dir: Path | None = None, embeddings=None):
    """Open the persisted Chroma collection.

    Raises :class:`IndexNotFoundError` with an actionable message rather than
    silently returning an empty collection - the prototype's failure mode was a
    confident answer drawn from nothing at all.
    """
    from langchain_chroma import Chroma

    from .providers import get_embeddings

    directory = Path(persist_dir or config.CHROMA_DIR)
    if not directory.exists() or not any(directory.iterdir()):
        raise IndexNotFoundError(
            f"No vector index at {directory}.\n"
            "Build one first:\n"
            "  python scripts/fetch_corpus.py\n"
            "  python -m rag.ingest"
        )

    store = Chroma(
        persist_directory=str(directory),
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
    )
    try:
        count = store._collection.count()
    except Exception as exc:  # pragma: no cover - defensive
        raise IndexNotFoundError(f"Could not open the Chroma collection at {directory}: {exc}") from exc

    if count == 0:
        raise IndexNotFoundError(
            f"The collection '{config.COLLECTION_NAME}' at {directory} is empty. "
            "Re-run: python -m rag.ingest --rebuild"
        )
    return store


def retrieve(
    question: str,
    store=None,
    k: int | None = None,
    min_relevance: float | None = None,
) -> list[tuple[Document, float]]:
    """Top-k passages above the relevance floor, best first.

    The floor is the first anti-hallucination gate: if nothing clears it the
    model is never asked the question, so it cannot invent an answer.
    """
    store = store if store is not None else load_vectorstore()
    top_k = k if k is not None else config.TOP_K
    floor = min_relevance if min_relevance is not None else config.MIN_RELEVANCE

    try:
        scored = store.similarity_search_with_relevance_scores(question, k=top_k)
    except Exception as exc:
        raise RuntimeError(f"Retrieval failed for question {question!r}: {exc}") from exc

    return [(doc, score) for doc, score in scored if score >= floor]


def answer_question(
    question: str,
    store=None,
    llm=None,
    k: int | None = None,
    min_relevance: float | None = None,
) -> AnswerBundle:
    """Retrieve, prompt, generate, and attach the sources behind the answer."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    from .providers import get_llm

    scored = retrieve(question, store=store, k=k, min_relevance=min_relevance)
    if not scored:
        return AnswerBundle(question=question, answer=config.REFUSAL_TEXT, refused=True)

    docs: Sequence[Document] = [doc for doc, _ in scored]
    blocks = build_blocks(docs)
    context = format_context(blocks)

    model = llm if llm is not None else get_llm()
    system = config.SYSTEM_PROMPT.format(refusal=config.REFUSAL_TEXT, context=context)

    try:
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=question)])
    except Exception as exc:
        raise RuntimeError(f"Answer generation failed for question {question!r}: {exc}") from exc

    text = response.content if hasattr(response, "content") else str(response)
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    refused = _is_refusal(text)
    return AnswerBundle(
        question=question,
        answer=text,
        sources=[b.source for b in blocks] if not refused else [],
        cited_numbers=extract_citation_numbers(text),
        refused=refused,
    )


def _is_refusal(text: str) -> bool:
    """Tolerant match against the configured refusal string."""
    normalized = " ".join(text.lower().split()).strip(' ."')
    target = " ".join(config.REFUSAL_TEXT.lower().split()).strip(' ."')
    return normalized == target or normalized.startswith(target)


def cited_sources(bundle: AnswerBundle):
    """The distinct (paper, page) locations the answer actually cited."""
    if bundle.refused:
        return []
    cited = [s for s in bundle.sources if not bundle.cited_numbers or s.number in bundle.cited_numbers]
    return dedupe_sources(cited)
