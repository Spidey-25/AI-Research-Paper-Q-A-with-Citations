"""Citations are the headline feature, so they get the most tests."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.citations import (
    AnswerBundle,
    build_blocks,
    dedupe_sources,
    extract_citation_numbers,
    format_context,
    has_citation,
    parse_context_blocks,
    source_from_document,
)


def test_source_carries_title_and_page_not_filename(sample_docs):
    source = source_from_document(sample_docs[0], 1)
    assert source.title == "Attention Is All You Need"
    assert source.arxiv_id == "1706.03762"
    assert source.page == 3
    # The regression this project exists to fix: never cite "pdf1.pdf".
    assert "pdf" not in source.render().lower().split("(")[0]


def test_source_falls_back_when_metadata_is_missing():
    bare = Document(page_content="text", metadata={"source": "mystery.pdf"})
    source = source_from_document(bare, 2)
    assert source.title == "mystery.pdf"
    assert source.arxiv_id == "n/a"
    assert source.page == 0


def test_source_survives_non_integer_page():
    doc = Document(page_content="t", metadata={"title": "T", "page_number": "not-a-number"})
    assert source_from_document(doc, 1).page == 0


def test_format_context_includes_metadata_for_every_block(sample_docs):
    context = format_context(build_blocks(sample_docs))
    assert "[1] Attention Is All You Need | arXiv:1706.03762 | page 3" in context
    assert "[2] BERT" in context
    assert "arXiv:1810.04805 | page 4" in context
    # The passage text must still be present - metadata is added, not substituted.
    assert "N=6 identical layers" in context


def test_context_format_round_trips(sample_docs):
    """format_context and parse_context_blocks must stay mutual inverses."""
    blocks = build_blocks(sample_docs)
    parsed = parse_context_blocks(format_context(blocks))
    assert [n for n, _, _ in parsed] == [1, 2]
    assert "N=6 identical layers" in parsed[0][2]
    assert "110M parameters" in parsed[1][2]


def test_parse_context_blocks_on_garbage_returns_nothing():
    assert parse_context_blocks("no numbered blocks here at all") == []


def test_extract_citation_numbers_handles_grouped_markers():
    assert extract_citation_numbers("Answer [1] and more [2].") == {1, 2}
    assert extract_citation_numbers("Combined [1, 3] form.") == {1, 3}
    assert extract_citation_numbers("Spaced [2 , 4].") == {2, 4}
    assert extract_citation_numbers("No citations here.") == set()


def test_has_citation():
    assert has_citation("The encoder has six layers [1].")
    assert not has_citation("The encoder has six layers.")
    # A bracketed non-number is not a citation.
    assert not has_citation("See [appendix] for detail.")


def test_dedupe_sources_collapses_same_paper_and_page(sample_docs):
    duplicated = [sample_docs[0], sample_docs[0], sample_docs[1]]
    sources = [b.source for b in build_blocks(duplicated)]
    deduped = dedupe_sources(sources)
    assert len(deduped) == 2
    assert deduped[0].number == 1  # first appearance wins


def test_render_appends_source_list(sample_docs):
    bundle = AnswerBundle(
        question="How many layers?",
        answer="Six identical layers [1].",
        sources=[b.source for b in build_blocks(sample_docs)],
        cited_numbers={1},
    )
    rendered = bundle.render()
    assert "Sources:" in rendered
    assert "Attention Is All You Need (arXiv:1706.03762), page 3" in rendered
    # Retrieved-but-uncited sources are shown, and labelled as such.
    assert "(retrieved, not cited)" in rendered


def test_render_of_a_refusal_has_no_source_list():
    bundle = AnswerBundle(question="q", answer="Nope.", refused=True)
    assert "Sources:" not in bundle.render()
