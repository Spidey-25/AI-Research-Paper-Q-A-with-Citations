"""End-to-end pipeline behaviour and the scoring logic of the eval harness."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from rag import config
from rag.pipeline import IndexNotFoundError, answer_question, cited_sources, load_vectorstore, retrieve


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def test_retrieval_returns_scored_documents(temp_store):
    results = retrieve("How many layers are in the encoder?", store=temp_store, k=2)
    assert results
    doc, score = results[0]
    assert 0.0 <= score <= 1.0
    assert doc.metadata["arxiv_id"] in {"1706.03762", "1810.04805"}


def test_retrieval_ranks_the_right_paper_first(temp_store):
    doc, _ = retrieve("How many parameters does BERT-LARGE have?", store=temp_store, k=2)[0]
    assert doc.metadata["arxiv_id"] == "1810.04805"


def test_relevance_floor_can_reject_everything(temp_store):
    assert retrieve("anything at all", store=temp_store, k=2, min_relevance=0.99) == []


def test_answer_is_grounded_and_cited(temp_store):
    bundle = answer_question("How many identical layers are in the encoder?", store=temp_store, k=2)
    assert not bundle.refused
    assert bundle.cited_numbers, "answer carried no inline citation"
    assert bundle.sources
    # Every marker must point at a passage that was actually retrieved.
    assert bundle.cited_numbers <= {s.number for s in bundle.sources}


def test_rendered_answer_shows_paper_and_page(temp_store):
    rendered = answer_question(
        "How many identical layers are in the encoder?", store=temp_store, k=2
    ).render()
    assert "Sources:" in rendered
    assert "arXiv:1706.03762" in rendered
    assert "page 3" in rendered


def test_out_of_corpus_question_is_refused(temp_store):
    bundle = answer_question("What is the capital city of Australia?", store=temp_store, k=2)
    assert bundle.refused
    assert bundle.answer == config.REFUSAL_TEXT
    assert bundle.sources == []


def test_refusal_when_relevance_floor_filters_everything(temp_store):
    """The floor must short-circuit before the model is ever asked."""
    bundle = answer_question("anything", store=temp_store, k=2, min_relevance=0.99)
    assert bundle.refused


def test_cited_sources_are_deduplicated(temp_store):
    bundle = answer_question("How many identical layers are in the encoder?", store=temp_store, k=2)
    sources = cited_sources(bundle)
    assert len({(s.arxiv_id, s.page) for s in sources}) == len(sources)


def test_empty_question_is_rejected(temp_store):
    with pytest.raises(ValueError):
        answer_question("   ", store=temp_store)


def test_missing_index_raises_actionable_error(tmp_path):
    with pytest.raises(IndexNotFoundError, match="ingest"):
        load_vectorstore(tmp_path / "does-not-exist")


# --------------------------------------------------------------------------
# Eval harness scoring
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def harness(repo_root=None):
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_eval", root / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eval"] = module
    spec.loader.exec_module(module)
    return module


def test_keyword_groups_require_one_form_from_each(harness):
    ok, missing = harness.keywords_matched(
        "The encoder is a stack of N=6 identical layers.", [["n=6", "six"], ["layers"]]
    )
    assert ok and not missing


def test_keyword_matching_reports_what_was_missing(harness):
    ok, missing = harness.keywords_matched("The encoder has some layers.", [["n=6", "six"], ["layers"]])
    assert not ok
    assert missing == ["n=6|six"]


def test_keyword_matching_is_case_and_whitespace_insensitive(harness):
    ok, _ = harness.keywords_matched("BERT-LARGE  has\n340M   parameters", [["340m"]])
    assert ok


def test_empty_keyword_set_passes(harness):
    assert harness.keywords_matched("anything", []) == (True, [])


def test_aggregate_computes_the_headline_metrics(harness):
    rows = [
        {"type": "in_corpus", "retrieval_hit": True, "answer_correct": True,
         "citation_ok": True, "refusal_ok": True, "refused": False, "elapsed_s": 0.1},
        {"type": "in_corpus", "retrieval_hit": False, "answer_correct": False,
         "citation_ok": False, "refusal_ok": False, "refused": True, "elapsed_s": 0.1},
        {"type": "out_of_corpus", "retrieval_hit": None, "answer_correct": True,
         "citation_ok": True, "refusal_ok": True, "refused": True, "elapsed_s": 0.1},
        {"type": "out_of_corpus", "retrieval_hit": None, "answer_correct": False,
         "citation_ok": False, "refusal_ok": False, "refused": False, "elapsed_s": 0.1},
    ]
    summary = harness.aggregate(rows)
    assert summary["retrieval_accuracy"] == 50.0
    assert summary["answer_correctness"] == 50.0
    assert summary["refusal_accuracy"] == 50.0
    assert summary["hallucinated"] == 1
    assert summary["false_refusals"] == 1


def test_testset_is_well_formed(harness, repo_root):
    data = json.loads((repo_root / "eval" / "testset.json").read_text())
    questions = data["questions"]
    assert 15 <= len(questions) <= 20

    ids = [q["id"] for q in questions]
    assert len(set(ids)) == len(ids), "duplicate question ids"

    out_of_corpus = [q for q in questions if q["type"] == "out_of_corpus"]
    assert 4 <= len(out_of_corpus) <= 5, "the anti-hallucination set must have 4-5 questions"

    for q in questions:
        assert q["question"].strip().endswith("?")
        assert q["expected_answer"].strip()
        if q["type"] == "in_corpus":
            assert q["expected_sources"], f"{q['id']} has no expected source"
            assert q["answer_must_contain"], f"{q['id']} has no gradeable keywords"
        else:
            assert q["expected_answer"] == "REFUSAL"
            assert not q["expected_sources"]


def test_every_expected_source_exists_in_the_manifest(repo_root):
    from rag.corpus import load_manifest

    known = {p.arxiv_id for p in load_manifest(repo_root / "corpus" / "manifest.json")}
    data = json.loads((repo_root / "eval" / "testset.json").read_text())
    for q in data["questions"]:
        for src in q.get("expected_sources", []):
            assert src in known, f"{q['id']} expects unknown paper {src}"
