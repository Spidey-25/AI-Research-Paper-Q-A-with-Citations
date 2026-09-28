"""Build the vector index from the PDF corpus.

Run once after fetching the corpus::

    python scripts/fetch_corpus.py
    python -m rag.ingest

Every chunk is stamped with the paper's title, arXiv id, authors, licence and
one-based page number *before* it is embedded, which is what allows the query
side to produce a real citation instead of a filename.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from langchain_core.documents import Document

from . import config
from .corpus import ManifestError, Paper, load_manifest, missing_pdfs


def scalar_metadata(metadata: dict) -> dict:
    """Keep only values Chroma can store.

    Chroma rejects nested or None metadata values. LangChain's
    ``filter_complex_metadata`` does this, but it lives in ``langchain_community``,
    which upstream has marked as sunset and unmaintained; carrying that whole
    package for one helper is not worth a deprecation warning on every run.
    """
    clean: dict = {}
    for key, value in metadata.items():
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            clean[key] = value
    return clean


def load_pdf(paper: Paper, pdf_dir: Path) -> list[Document]:
    """Load one PDF, one Document per page, with citation metadata attached."""
    from pypdf import PdfReader

    path = pdf_dir / paper.filename
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}: {exc}\n"
            "The file may be truncated - re-run: python scripts/fetch_corpus.py --force "
            f"--only {paper.arxiv_id}"
        ) from exc

    enriched: list[Document] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a single malformed page must not kill the run
            print(f"    warning: page {index + 1} of {paper.filename} failed to parse ({exc})")
            continue
        if not text.strip():
            continue
        metadata = paper.as_metadata()
        # Page numbers are one-based, because that is what a citation means.
        metadata["page_number"] = index + 1
        enriched.append(Document(page_content=text, metadata=metadata))

    if not enriched:
        raise RuntimeError(
            f"{path} produced no extractable text. If it is a scanned image PDF it needs "
            "OCR before it can be indexed."
        )
    return enriched


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split pages into embedding-sized chunks, keeping page metadata intact."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata = scalar_metadata(chunk.metadata)
    return [c for c in chunks if c.page_content.strip()]


def build_index(
    pdf_dir: Path,
    persist_dir: Path,
    manifest_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
    verbose: bool = True,
) -> int:
    """Ingest the corpus and persist the vector store. Returns the chunk count."""
    from langchain_chroma import Chroma

    from .providers import get_embeddings

    papers = load_manifest(manifest_path)
    absent = missing_pdfs(papers, pdf_dir)
    if absent:
        names = ", ".join(p.arxiv_id for p in absent)
        raise FileNotFoundError(
            f"{len(absent)} corpus PDF(s) missing from {pdf_dir}: {names}\n"
            "The PDFs are not committed to this repository by design. Fetch them:\n"
            "  python scripts/fetch_corpus.py"
        )

    if persist_dir.exists() and any(persist_dir.iterdir()):
        if not rebuild:
            raise FileExistsError(
                f"A vector index already exists at {persist_dir}.\n"
                "Pass --rebuild to discard and rebuild it."
            )
        if verbose:
            print(f"Removing existing index at {persist_dir}")
        shutil.rmtree(persist_dir)

    all_chunks: list[Document] = []
    for paper in papers:
        pages = load_pdf(paper, pdf_dir)
        chunks = chunk_documents(pages, chunk_size, chunk_overlap)
        if verbose:
            print(f"  {paper.arxiv_id}  {len(pages):>3} pages -> {len(chunks):>4} chunks  {paper.title[:50]}")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks were produced from the corpus - nothing to index.")

    if verbose:
        print(f"\nEmbedding {len(all_chunks)} chunks with the "
              f"'{config.EMBEDDING_PROVIDER}' provider...")

    persist_dir.mkdir(parents=True, exist_ok=True)
    try:
        Chroma.from_documents(
            documents=all_chunks,
            embedding=get_embeddings(),
            persist_directory=str(persist_dir),
            collection_name=config.COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build the vector store at {persist_dir}: {exc}") from exc

    if verbose:
        print(f"Index written to {persist_dir} (collection '{config.COLLECTION_NAME}').")
    return len(all_chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest",
        description="Build the Chroma vector index from the arXiv corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdf-dir", type=Path, default=config.PDF_DIR)
    parser.add_argument("--persist-dir", type=Path, default=config.CHROMA_DIR)
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--rebuild", action="store_true", help="discard any existing index first")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.chunk_overlap >= args.chunk_size:
        print("ERROR: --chunk-overlap must be smaller than --chunk-size", file=sys.stderr)
        return 2

    verbose = not args.quiet
    if verbose:
        print(f"Embedding provider: {config.EMBEDDING_PROVIDER}")
        print(f"Corpus: {args.pdf_dir}\n")

    try:
        count = build_index(
            pdf_dir=args.pdf_dir,
            persist_dir=args.persist_dir,
            manifest_path=args.manifest,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            rebuild=args.rebuild,
            verbose=verbose,
        )
    except (ManifestError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if verbose:
        print(f"\nDone: {count} chunks indexed. Ask a question:\n"
              '  python -m rag.query "What is scaled dot-product attention?"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
