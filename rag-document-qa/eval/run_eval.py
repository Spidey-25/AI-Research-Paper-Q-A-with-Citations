#!/usr/bin/env python3
"""Evaluate the RAG pipeline against the hand-built test set.

Four things are measured, because "the answer looked good" is not a metric:

1. Retrieval accuracy   - did the paper that actually contains the answer appear
                          in the top-k passages? Everything downstream is capped
                          by this number.
2. Answer correctness   - does the answer contain the facts the hand-written
                          gold answer requires? Containment matching, not an LLM
                          judge, so the score is deterministic and free.
3. Citation rate        - did the answer carry at least one inline [n] marker,
                          and did every marker point at a passage that was
                          actually retrieved? A citation to passage [9] when only
                          5 were retrieved is a fabricated citation.
4. Refusal accuracy     - on the 5 deliberately out-of-corpus questions, did the
                          system emit the refusal string instead of inventing an
                          answer? This is the anti-hallucination measurement.

Usage::

    python eval/run_eval.py
    python eval/run_eval.py --k 8 --json results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag import config  # noqa: E402
from rag.citations import extract_citation_numbers  # noqa: E402
from rag.pipeline import IndexNotFoundError, answer_question, load_vectorstore, retrieve  # noqa: E402

DEFAULT_TESTSET = Path(__file__).resolve().parent / "testset.json"


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, and pad so ' 6 ' style forms can match."""
    return " " + re.sub(r"\s+", " ", text.lower()).strip() + " "


def keywords_matched(answer: str, groups: list[list[str]]) -> tuple[bool, list[str]]:
    """Every group must contribute at least one surface form. Returns misses."""
    if not groups:
        return True, []
    haystack = normalise(answer)
    missing: list[str] = []
    for group in groups:
        if not any(normalise(form).strip() in haystack for form in group):
            missing.append("|".join(group))
    return not missing, missing


def load_testset(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Test set not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not data.get("questions"):
        raise ValueError(f"{path} contains no questions")
    return data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_one(case: dict, store, k: int, min_relevance: float) -> dict:
    """Score a single question. Never raises: a crash is recorded as a failure."""
    question = case["question"]
    is_out_of_corpus = case.get("type") == "out_of_corpus"
    row: dict = {
        "id": case.get("id", "?"),
        "type": case.get("type", "in_corpus"),
        "question": question,
        "error": None,
    }

    started = time.perf_counter()
    try:
        scored = retrieve(question, store=store, k=k, min_relevance=min_relevance)
        retrieved_ids = [doc.metadata.get("arxiv_id", "?") for doc, _ in scored]
        bundle = answer_question(question, store=store, k=k, min_relevance=min_relevance)
    except Exception as exc:
        row.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "retrieval_hit": False,
                "answer_correct": False,
                "citation_ok": False,
                "refusal_ok": False,
                "answer": "",
                "retrieved": [],
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
        )
        return row

    expected_sources = case.get("expected_sources") or []
    retrieval_hit = (
        all(src in retrieved_ids for src in expected_sources) if expected_sources else None
    )

    answer = bundle.answer
    cited = extract_citation_numbers(answer)
    valid_range = set(range(1, len(scored) + 1))
    # A marker outside the retrieved range is a fabricated citation.
    fabricated = sorted(cited - valid_range)

    if is_out_of_corpus:
        refusal_ok = bundle.refused
        answer_correct = bundle.refused
        # Refusals must not carry citations - there is nothing to cite.
        citation_ok = not cited if bundle.refused else False
        missing: list[str] = []
    else:
        refusal_ok = not bundle.refused
        answer_correct, missing = keywords_matched(answer, case.get("answer_must_contain", []))
        answer_correct = answer_correct and not bundle.refused
        citation_ok = bool(cited) and not fabricated

    row.update(
        {
            "answer": answer,
            "retrieved": retrieved_ids,
            "expected_sources": expected_sources,
            "retrieval_hit": retrieval_hit,
            "answer_correct": answer_correct,
            "citation_ok": citation_ok,
            "refused": bundle.refused,
            "refusal_ok": refusal_ok,
            "cited": sorted(cited),
            "fabricated_citations": fabricated,
            "missing_keywords": missing,
            "elapsed_s": round(time.perf_counter() - started, 3),
        }
    )
    return row


def aggregate(rows: list[dict]) -> dict:
    in_corpus = [r for r in rows if r["type"] == "in_corpus"]
    out_corpus = [r for r in rows if r["type"] == "out_of_corpus"]

    def pct(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / denominator, 1) if denominator else 0.0

    retrieval_scored = [r for r in in_corpus if r.get("retrieval_hit") is not None]
    hallucinated = [r for r in out_corpus if not r.get("refused")]

    return {
        "total": len(rows),
        "in_corpus": len(in_corpus),
        "out_of_corpus": len(out_corpus),
        "errors": sum(1 for r in rows if r.get("error")),
        "retrieval_accuracy": pct(sum(1 for r in retrieval_scored if r["retrieval_hit"]), len(retrieval_scored)),
        "retrieval_hits": sum(1 for r in retrieval_scored if r["retrieval_hit"]),
        "retrieval_total": len(retrieval_scored),
        "answer_correctness": pct(sum(1 for r in in_corpus if r["answer_correct"]), len(in_corpus)),
        "answer_correct_count": sum(1 for r in in_corpus if r["answer_correct"]),
        "citation_rate": pct(sum(1 for r in in_corpus if r["citation_ok"]), len(in_corpus)),
        "citation_count": sum(1 for r in in_corpus if r["citation_ok"]),
        "fabricated_citations": sum(len(r.get("fabricated_citations") or []) for r in rows),
        "refusal_accuracy": pct(sum(1 for r in out_corpus if r["refusal_ok"]), len(out_corpus)),
        "refusal_count": sum(1 for r in out_corpus if r["refusal_ok"]),
        "hallucinated": len(hallucinated),
        "false_refusals": sum(1 for r in in_corpus if r.get("refused")),
        "mean_latency_s": round(sum(r.get("elapsed_s", 0.0) for r in rows) / len(rows), 3) if rows else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def tick(value) -> str:
    if value is None:
        return " - "
    return " Y " if value else " N "


def print_table(rows: list[dict]) -> None:
    header = f"{'ID':<5} {'TYPE':<14} {'RETR':^6} {'ANS':^5} {'CITE':^6} {'REFUSE':^7}  QUESTION"
    print(header)
    print("-" * len(header))
    for r in rows:
        kind = "out-of-corpus" if r["type"] == "out_of_corpus" else "in-corpus"
        question = r["question"]
        if len(question) > 58:
            question = question[:55] + "..."
        print(
            f"{r['id']:<5} {kind:<14} {tick(r.get('retrieval_hit')):^6} "
            f"{tick(r.get('answer_correct')):^5} {tick(r.get('citation_ok')):^6} "
            f"{tick(r.get('refusal_ok')):^7}  {question}"
        )


def print_failures(rows: list[dict]) -> None:
    failures = [
        r
        for r in rows
        if r.get("error")
        or not r.get("answer_correct")
        or not r.get("citation_ok")
        or r.get("retrieval_hit") is False
    ]
    if not failures:
        return
    print("\nFailure detail")
    print("-" * 78)
    for r in failures:
        print(f"{r['id']}  {r['question']}")
        if r.get("error"):
            print(f"     error: {r['error']}")
            continue
        if r.get("retrieval_hit") is False:
            print(f"     retrieval: expected {r['expected_sources']}, got {r['retrieved']}")
        if r.get("missing_keywords"):
            print(f"     missing:   {', '.join(r['missing_keywords'])}")
        if r.get("fabricated_citations"):
            print(f"     fabricated citations: {r['fabricated_citations']}")
        if not r.get("citation_ok") and not r.get("fabricated_citations"):
            print("     citation:  no inline [n] marker in the answer")
        answer = " ".join(r.get("answer", "").split())
        print(f"     answer:    {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print()


def print_summary(summary: dict) -> None:
    print("\nAggregate scores")
    print("-" * 78)
    print(
        f"  Retrieval accuracy   {summary['retrieval_accuracy']:>6.1f}%   "
        f"({summary['retrieval_hits']}/{summary['retrieval_total']} in-corpus questions "
        "retrieved the correct paper)"
    )
    print(
        f"  Answer correctness   {summary['answer_correctness']:>6.1f}%   "
        f"({summary['answer_correct_count']}/{summary['in_corpus']} contained every required fact)"
    )
    print(
        f"  Citation rate        {summary['citation_rate']:>6.1f}%   "
        f"({summary['citation_count']}/{summary['in_corpus']} answers carried a valid inline citation)"
    )
    print(
        f"  Refusal accuracy     {summary['refusal_accuracy']:>6.1f}%   "
        f"({summary['refusal_count']}/{summary['out_of_corpus']} out-of-corpus questions correctly refused)"
    )
    print()
    print(f"  Hallucinated answers      {summary['hallucinated']}  (out-of-corpus questions answered anyway)")
    print(f"  False refusals            {summary['false_refusals']}  (in-corpus questions wrongly refused)")
    print(f"  Fabricated citations      {summary['fabricated_citations']}  (markers pointing outside the retrieved set)")
    print(f"  Errors                    {summary['errors']}")
    print(f"  Mean latency              {summary['mean_latency_s']}s per question")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the RAG pipeline against eval/testset.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--persist-dir", type=Path, default=config.CHROMA_DIR)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--min-relevance", type=float, default=config.MIN_RELEVANCE)
    parser.add_argument("--json", type=Path, help="also write full results to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="summary only, no per-question table")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero if answer correctness falls below this percentage (for CI)",
    )
    args = parser.parse_args(argv)

    try:
        testset = load_testset(args.testset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        store = load_vectorstore(args.persist_dir)
    except IndexNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("RAG evaluation")
    print("=" * 78)
    print(f"  test set            {args.testset}")
    print(f"  embedding provider  {config.EMBEDDING_PROVIDER}")
    print(f"  llm provider        {config.LLM_PROVIDER}")
    print(f"  top-k               {args.k}")
    print(f"  min relevance       {args.min_relevance}")
    print(f"  refusal string      {config.REFUSAL_TEXT!r}")
    print()

    rows = [evaluate_one(case, store, args.k, args.min_relevance) for case in testset["questions"]]
    summary = aggregate(rows)

    if not args.quiet:
        print_table(rows)
        print_failures(rows)
    print_summary(summary)

    if args.json:
        payload = {
            "config": {
                "embedding_provider": config.EMBEDDING_PROVIDER,
                "llm_provider": config.LLM_PROVIDER,
                "k": args.k,
                "min_relevance": args.min_relevance,
                "testset": str(args.testset),
            },
            "summary": summary,
            "results": rows,
        }
        try:
            args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nFull results written to {args.json}")
        except OSError as exc:
            print(f"WARNING: could not write {args.json}: {exc}", file=sys.stderr)

    if args.fail_under is not None and summary["answer_correctness"] < args.fail_under:
        print(
            f"\nFAIL: answer correctness {summary['answer_correctness']}% "
            f"is below the --fail-under threshold of {args.fail_under}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
