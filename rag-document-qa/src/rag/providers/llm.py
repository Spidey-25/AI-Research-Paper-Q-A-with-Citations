"""Answer-generation providers.

``anthropic``
    The production path: a Claude model via ``langchain-anthropic``. Needs
    ``ANTHROPIC_API_KEY``.

``offline``
    :class:`OfflineExtractiveLLM` - a deterministic reader that answers only by
    quoting the retrieved passages, cites them, and refuses when the passages do
    not cover the question. It has no weights and cannot paraphrase, so it will
    never score as well as Claude on answer quality. What it *can* do is
    exercise the entire pipeline - retrieval, citation formatting, the refusal
    contract and the eval harness - with zero credentials, which is what keeps
    this repository verifiable by anyone who clones it.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from .. import config
from ..citations import parse_context_blocks
from .embeddings import tokenize


class LLMProviderError(RuntimeError):
    """Raised when a generation backend cannot be reached or configured."""


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> List[str]:
    """Cheap sentence splitter - good enough for extractive quoting."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text.replace("\n", " ")) if p.strip()]
    return [p for p in parts if len(p) > 2]


class OfflineExtractiveLLM(SimpleChatModel):
    """Deterministic extractive reader used when no API key is available.

    The contract it implements is exactly the contract the system prompt asks a
    real model for:

    * answer only from the numbered context passages,
    * cite the passage number inline,
    * emit the refusal string verbatim when the context does not cover the
      question.
    """

    #: Minimum IDF-weighted share of the question's content words that must
    #: appear in a passage before that passage is considered to answer the
    #: question. Configurable via RAG_OFFLINE_COVERAGE.
    coverage_threshold: float = config.OFFLINE_COVERAGE_THRESHOLD

    #: Absolute floor: at least this many distinct question words must match.
    min_matched_terms: int = 2

    #: How many passages may contribute sentences to one answer.
    max_blocks: int = 2

    #: How many sentences may be quoted from each contributing passage.
    max_sentences_per_block: int = 2

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _split_prompt(messages: List[BaseMessage]) -> tuple[str, str]:
        """Return (system_text, question_text) from a chat message list."""
        system_parts: List[str] = []
        user_parts: List[str] = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.type == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        return "\n".join(system_parts), "\n".join(user_parts)

    @staticmethod
    def _token_weights(question_tokens: set[str], blocks: List[str]) -> dict[str, float]:
        """Inverse-document-frequency weights over the retrieved passages.

        A word that occurs in every retrieved passage ("training", "model") says
        almost nothing about whether the question is answerable. A word that
        occurs in none of them ("ImageNet", "schedule") is precisely the signal
        that it is not. Weighting coverage this way is what stops the reader
        answering an out-of-corpus question purely on shared jargon.
        """
        n = max(len(blocks), 1)
        block_tokens = [set(tokenize(b)) for b in blocks]
        weights: dict[str, float] = {}
        for token in question_tokens:
            df = sum(1 for bt in block_tokens if token in bt)
            weights[token] = math.log(1.0 + n / (df + 0.5))
        return weights

    @staticmethod
    def _coverage(
        question_tokens: set[str],
        text: str,
        weights: dict[str, float] | None = None,
    ) -> tuple[float, int]:
        """Weighted share of the question's content words present in ``text``."""
        if not question_tokens:
            return 0.0, 0
        matched = question_tokens & set(tokenize(text))
        if not weights:
            return len(matched) / len(question_tokens), len(matched)
        total = sum(weights.get(t, 1.0) for t in question_tokens)
        if total <= 0:
            return 0.0, len(matched)
        hit = sum(weights.get(t, 1.0) for t in matched)
        return hit / total, len(matched)

    # -- generation ------------------------------------------------------

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        system_text, question = self._split_prompt(messages)
        blocks = parse_context_blocks(system_text)
        if not blocks:
            return config.REFUSAL_TEXT

        q_tokens = set(tokenize(question))
        weights = self._token_weights(q_tokens, [body for _, _, body in blocks])

        scored = []
        for number, _header, body in blocks:
            coverage, matched = self._coverage(q_tokens, body, weights)
            scored.append((coverage, matched, number, body))
        scored.sort(key=lambda row: (-row[0], row[2]))

        best_coverage, best_matched, _, _ = scored[0]
        if best_coverage < self.coverage_threshold or best_matched < self.min_matched_terms:
            return config.REFUSAL_TEXT

        fragments: List[str] = []
        for coverage, _matched, number, body in scored[: self.max_blocks]:
            if coverage < self.coverage_threshold:
                continue
            sentences = split_sentences(body)
            if not sentences:
                continue
            ranked = sorted(
                sentences,
                key=lambda s: (-self._coverage(q_tokens, s, weights)[0], len(s)),
            )
            chosen = [s for s in ranked[: self.max_sentences_per_block] if self._coverage(q_tokens, s, weights)[1] > 0]
            if not chosen:
                chosen = ranked[:1]
            # Restore the passage's own sentence order for readability.
            chosen.sort(key=sentences.index)
            quoted = " ".join(chosen).strip()
            fragments.append(f"{quoted} [{number}]")

        if not fragments:
            return config.REFUSAL_TEXT
        return " ".join(fragments)


def _build_anthropic():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise LLMProviderError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set RAG_LLM_PROVIDER=offline to run without any credentials."
        )
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - import guard
        raise LLMProviderError(
            "langchain-anthropic is not installed. Install requirements.txt, or set "
            "RAG_LLM_PROVIDER=offline."
        ) from exc

    try:
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=0,
        )
    except Exception as exc:
        raise LLMProviderError(
            f"Could not construct ChatAnthropic with model '{config.ANTHROPIC_MODEL}': {exc}\n"
            "Model ids change over time - set RAG_LLM_MODEL to a current one "
            "(see https://platform.claude.com/docs/en/about-claude/models/overview)."
        ) from exc


def get_llm(provider: str | None = None):
    """Return the configured chat model.

    Selected by ``RAG_LLM_PROVIDER`` (``anthropic`` | ``offline``).
    """
    name = (provider or config.LLM_PROVIDER).lower()
    if name == "anthropic":
        return _build_anthropic()
    if name == "offline":
        return OfflineExtractiveLLM()
    raise LLMProviderError(
        f"Unknown LLM provider {name!r}. Expected 'anthropic' or 'offline'."
    )
