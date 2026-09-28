"""The corpus manifest is a licence-compliance artefact as well as config,
so it is validated rather than trusted."""

from __future__ import annotations

import json
import re

import pytest
from langchain_core.documents import Document

from rag.corpus import ManifestError, Paper, by_filename, load_manifest, missing_pdfs
from rag.ingest import chunk_documents, scalar_metadata

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def test_manifest_loads_and_has_a_real_corpus(repo_root):
    papers = load_manifest(repo_root / "corpus" / "manifest.json")
    assert 5 <= len(papers) <= 6


def test_every_paper_is_fully_attributed(repo_root):
    for paper in load_manifest(repo_root / "corpus" / "manifest.json"):
        assert ARXIV_ID_RE.match(paper.arxiv_id), paper.arxiv_id
        assert paper.title.strip()
        assert paper.authors, f"{paper.arxiv_id} has no authors"
        assert paper.licence.strip(), f"{paper.arxiv_id} has no licence"
        assert paper.licence_url.startswith("http")
        assert paper.pdf_url.startswith("https://arxiv.org/")
        assert paper.filename.endswith(".pdf")


def test_no_cricket_documents_remain(repo_root):
    """Regression guard: the original corpus was three copyright-encumbered
    cricket PDFs. None of them may ever come back."""
    text = (repo_root / "corpus" / "manifest.json").read_text().lower()
    for banned in ("cricket", "mcc", "marylebone", "pdf1.pdf", "pdf2.pdf", "pdf3.pdf"):
        assert banned not in text, f"{banned!r} found in the manifest"


def test_pdfs_are_not_committed(repo_root):
    """Redistribution guard: the repository must never carry the PDFs."""
    committed = list((repo_root / "corpus").glob("*.pdf"))
    assert not committed, f"PDFs must not be committed: {committed}"


def test_filenames_and_ids_are_unique(repo_root):
    papers = load_manifest(repo_root / "corpus" / "manifest.json")
    assert len({p.filename for p in papers}) == len(papers)
    assert len({p.arxiv_id for p in papers}) == len(papers)


def test_by_filename_indexes_the_corpus(repo_root):
    papers = load_manifest(repo_root / "corpus" / "manifest.json")
    index = by_filename(papers)
    assert index["1706.03762.pdf"].title == "Attention Is All You Need"


def test_missing_pdfs_reports_everything_when_directory_is_empty(repo_root, tmp_path):
    papers = load_manifest(repo_root / "corpus" / "manifest.json")
    assert len(missing_pdfs(papers, tmp_path)) == len(papers)


def test_missing_manifest_raises_a_useful_error(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")


def test_malformed_manifest_is_rejected(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text("{ not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(bad)


def test_manifest_missing_required_field_is_rejected(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"papers": [{"arxiv_id": "1234.5678", "title": "T"}]}))
    with pytest.raises(ManifestError, match="missing required field"):
        load_manifest(bad)


def test_duplicate_arxiv_id_is_rejected(tmp_path):
    entry = {
        "arxiv_id": "1234.5678",
        "title": "T",
        "filename": "a.pdf",
        "licence": "L",
        "pdf_url": "https://arxiv.org/pdf/1234.5678",
    }
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"papers": [entry, dict(entry, filename="b.pdf")]}))
    with pytest.raises(ManifestError, match="Duplicate"):
        load_manifest(bad)


def test_paper_metadata_is_chroma_safe():
    paper = Paper(
        short_name="x", arxiv_id="1234.5678", version="v1", title="T",
        authors=("A", "B"), year=2020, licence="L", licence_url="u",
        abs_url="a", pdf_url="p", filename="f.pdf",
    )
    for value in paper.as_metadata().values():
        assert isinstance(value, (str, int, float, bool))
    assert paper.as_metadata()["authors"] == "A, B"


# --------------------------------------------------------------------------
# Ingest helpers
# --------------------------------------------------------------------------

def test_scalar_metadata_drops_unstorable_values():
    cleaned = scalar_metadata(
        {"title": "T", "page_number": 3, "ok": True, "ratio": 1.5, "bad": ["a"], "none": None}
    )
    assert cleaned == {"title": "T", "page_number": 3, "ok": True, "ratio": 1.5}


def test_chunking_preserves_citation_metadata_on_every_chunk():
    """If a chunk loses its page number, its citation becomes a lie."""
    long_page = Document(
        page_content=("The Transformer uses multi-head attention. " * 80),
        metadata={"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "page_number": 4},
    )
    chunks = chunk_documents([long_page], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["page_number"] == 4
        assert chunk.metadata["arxiv_id"] == "1706.03762"


def test_chunking_drops_blank_chunks():
    blank = Document(page_content="   \n  ", metadata={"page_number": 1})
    assert chunk_documents([blank], chunk_size=300, chunk_overlap=50) == []
