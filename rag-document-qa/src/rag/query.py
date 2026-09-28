"""Ask a question against the indexed corpus.

    python -m rag.query "How many layers does the Transformer encoder have?"
    python -m rag.query "What is LoRA?" --k 8 --show-context
    python -m rag.query "What is BERT?" --json

The prototype hardcoded a single question at the bottom of the script; this is
an argparse CLI with a JSON mode so it can be driven by other tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .citations import AnswerBundle
from .pipeline import IndexNotFoundError, answer_question, cited_sources, load_vectorstore, retrieve


def bundle_to_dict(bundle: AnswerBundle) -> dict:
    return {
        "question": bundle.question,
        "answer": bundle.answer,
        "refused": bundle.refused,
        "cited": sorted(bundle.cited_numbers),
        "sources": [
            {
                "n": s.number,
                "title": s.title,
                "arxiv_id": s.arxiv_id,
                "page": s.page,
                "url": s.url,
                "cited": s.number in bundle.cited_numbers,
            }
            for s in bundle.sources
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag.query",
        description="Ask a question about the indexed arXiv corpus, with citations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="passages to retrieve")
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=config.MIN_RELEVANCE,
        help="discard passages scoring below this (0-1); refuse if none survive",
    )
    parser.add_argument("--persist-dir", type=Path, default=config.CHROMA_DIR)
    parser.add_argument("--show-context", action="store_true", help="print the retrieved passages")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    if not question:
        print("ERROR: empty question", file=sys.stderr)
        return 2
    if args.k < 1:
        print("ERROR: --k must be >= 1", file=sys.stderr)
        return 2

    try:
        store = load_vectorstore(args.persist_dir)
    except IndexNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: could not open the vector store: {exc}", file=sys.stderr)
        return 1

    if args.show_context:
        try:
            scored = retrieve(question, store=store, k=args.k, min_relevance=args.min_relevance)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print("Retrieved passages")
        print("-" * 78)
        for i, (doc, score) in enumerate(scored, start=1):
            meta = doc.metadata
            print(f"[{i}] score={score:.3f}  {meta.get('title', '?')} p.{meta.get('page_number', '?')}")
            print(f"    {doc.page_content[:220].strip()}...")
        print("-" * 78)
        print()

    try:
        bundle = answer_question(
            question, store=store, k=args.k, min_relevance=args.min_relevance
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(bundle_to_dict(bundle), indent=2))
        return 0

    print(bundle.render())
    if not bundle.refused and not bundle.cited_numbers:
        print("\nWarning: the model returned an answer without inline citations.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
