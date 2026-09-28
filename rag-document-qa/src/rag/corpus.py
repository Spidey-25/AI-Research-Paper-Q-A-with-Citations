"""Corpus manifest: the mapping from a PDF on disk to a real citation.

The manifest is the reason a citation in this project reads as
``Attention Is All You Need (arXiv:1706.03762), page 4`` rather than
``pdf1.pdf``. Metadata is attached at ingest time, so it is already in the
vector store by the time anything is retrieved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import config


class ManifestError(RuntimeError):
    """Raised when the corpus manifest is missing, malformed or incomplete."""


REQUIRED_FIELDS = ("arxiv_id", "title", "filename", "licence", "pdf_url")


@dataclass(frozen=True)
class Paper:
    short_name: str
    arxiv_id: str
    version: str
    title: str
    authors: tuple[str, ...]
    year: int
    licence: str
    licence_url: str
    abs_url: str
    pdf_url: str
    filename: str

    @property
    def author_string(self) -> str:
        return ", ".join(self.authors)

    def as_metadata(self) -> dict[str, str]:
        """Flat, scalar-only metadata - Chroma rejects nested values."""
        return {
            "source": self.filename,
            "title": self.title,
            "arxiv_id": self.arxiv_id,
            "short_name": self.short_name,
            "authors": self.author_string,
            "year": str(self.year),
            "licence": self.licence,
            "url": self.abs_url,
        }


def load_manifest(path: Path | str | None = None) -> list[Paper]:
    """Read and validate ``corpus/manifest.json``."""
    manifest_path = Path(path) if path else config.MANIFEST_PATH
    if not manifest_path.exists():
        raise ManifestError(
            f"Corpus manifest not found at {manifest_path}. "
            "It ships with the repository - check you are on a clean checkout."
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path} is not valid JSON: {exc}") from exc

    entries = raw.get("papers")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{manifest_path} contains no 'papers' list.")

    papers: list[Paper] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
        if missing:
            raise ManifestError(
                f"Paper #{i + 1} in {manifest_path} is missing required field(s): "
                f"{', '.join(missing)}"
            )
        if entry["arxiv_id"] in seen_ids:
            raise ManifestError(f"Duplicate arXiv id {entry['arxiv_id']} in {manifest_path}.")
        seen_ids.add(entry["arxiv_id"])
        papers.append(
            Paper(
                short_name=entry.get("short_name", entry["arxiv_id"]),
                arxiv_id=entry["arxiv_id"],
                version=entry.get("version", ""),
                title=entry["title"],
                authors=tuple(entry.get("authors", [])),
                year=int(entry.get("year", 0)),
                licence=entry["licence"],
                licence_url=entry.get("licence_url", ""),
                abs_url=entry.get("abs_url", f"https://arxiv.org/abs/{entry['arxiv_id']}"),
                pdf_url=entry["pdf_url"],
                filename=entry["filename"],
            )
        )
    return papers


def by_filename(papers: Iterable[Paper]) -> dict[str, Paper]:
    """Index papers by the PDF filename they are stored under."""
    return {p.filename: p for p in papers}


def missing_pdfs(papers: Iterable[Paper], pdf_dir: Path | None = None) -> list[Paper]:
    """Papers listed in the manifest whose PDF is not on disk yet."""
    directory = pdf_dir or config.PDF_DIR
    return [p for p in papers if not (directory / p.filename).exists()]
