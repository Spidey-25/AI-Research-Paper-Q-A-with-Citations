"""Citation plumbing: carry document metadata from retrieval into the prompt,
and back out into a rendered source list.

The original prototype threw this away::

    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

Every citation in this project flows through the single format defined here, so
the prompt builder, the offline reader model and the evaluation harness can
never drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from langchain_core.documents import Document

#: Header line of a context block, e.g.
#: ``[2] Attention Is All You Need | arXiv:1706.03762 | page 4``
BLOCK_HEADER_RE = re.compile(
    r"^\[(?P<n>\d+)\]\s+(?P<title>.+?)\s+\|\s+arXiv:(?P<arxiv_id>[^\s|]+)\s+\|\s+page\s+(?P<page>\d+)\s*$",
    re.MULTILINE,
)

#: Inline citation marker in an answer, e.g. ``[2]`` or ``[1, 3]``.
INLINE_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Source:
    """One citable location: a paper plus a page."""

    number: int
    title: str
    arxiv_id: str
    page: int
    source_file: str = ""
    url: str = ""

    @property
    def label(self) -> str:
        return f"{self.title} | arXiv:{self.arxiv_id} | page {self.page}"

    def render(self) -> str:
        tail = f" - {self.url}" if self.url else ""
        return f"[{self.number}] {self.title} (arXiv:{self.arxiv_id}), page {self.page}{tail}"


@dataclass
class ContextBlock:
    """A numbered passage handed to the model."""

    source: Source
    text: str


@dataclass
class AnswerBundle:
    """An answer plus the sources that back it."""

    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    cited_numbers: set[int] = field(default_factory=set)
    refused: bool = False

    def render(self) -> str:
        """Human-readable answer with its source list appended."""
        if self.refused or not self.sources:
            return self.answer
        lines = [self.answer, "", "Sources:"]
        cited = self.cited_numbers
        for src in self.sources:
            marker = "" if not cited or src.number in cited else "   (retrieved, not cited)"
            lines.append(f"  {src.render()}{marker}")
        return "\n".join(lines)


def _meta(doc: Document, key: str, default: str = "") -> str:
    value = doc.metadata.get(key, default)
    return default if value is None else str(value)


def source_from_document(doc: Document, number: int) -> Source:
    """Build a :class:`Source` from a retrieved document's metadata.

    Falls back gracefully when a document predates metadata enrichment, so a
    stale index degrades to a filename citation rather than crashing.
    """
    title = _meta(doc, "title") or _meta(doc, "source") or "unknown document"
    arxiv_id = _meta(doc, "arxiv_id") or "n/a"
    raw_page = doc.metadata.get("page_number", doc.metadata.get("page", 0))
    try:
        page = int(raw_page)
    except (TypeError, ValueError):
        page = 0
    return Source(
        number=number,
        title=title,
        arxiv_id=arxiv_id,
        page=page,
        source_file=_meta(doc, "source"),
        url=_meta(doc, "url"),
    )


def build_blocks(docs: Sequence[Document]) -> list[ContextBlock]:
    """Number the retrieved documents and pair each with its citable source."""
    return [
        ContextBlock(source=source_from_document(doc, i), text=doc.page_content.strip())
        for i, doc in enumerate(docs, start=1)
    ]


def format_context(blocks: Iterable[ContextBlock]) -> str:
    """Render numbered blocks for the prompt.

    This replaces the metadata-destroying ``format_docs`` of the prototype: the
    model now sees which paper and which page each passage came from, which is
    what makes an inline citation possible at all.
    """
    return "\n\n".join(f"[{b.source.number}] {b.source.label}\n{b.text}" for b in blocks)


def parse_context_blocks(context: str) -> list[tuple[int, str, str]]:
    """Inverse of :func:`format_context`: ``(number, header, body)`` per block.

    Used by the offline reader model, which receives only the rendered prompt
    and must recover the passages from it.
    """
    matches = list(BLOCK_HEADER_RE.finditer(context))
    out: list[tuple[int, str, str]] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        out.append((int(m.group("n")), m.group(0).strip(), context[body_start:body_end].strip()))
    return out


def extract_citation_numbers(answer: str) -> set[int]:
    """Every distinct ``[n]`` referenced by an answer, including ``[1, 3]`` forms."""
    numbers: set[int] = set()
    for group in INLINE_CITATION_RE.findall(answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                numbers.add(int(part))
    return numbers


def has_citation(answer: str) -> bool:
    """True when the answer carries at least one inline citation marker."""
    return bool(extract_citation_numbers(answer))


def dedupe_sources(sources: Sequence[Source]) -> list[Source]:
    """Collapse repeats of the same (paper, page), keeping first appearance order."""
    seen: set[tuple[str, int]] = set()
    out: list[Source] = []
    for src in sources:
        key = (src.arxiv_id, src.page)
        if key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out
