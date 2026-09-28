"""Provider interfaces: determinism, batching, and the refusal contract."""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from rag import config
from rag.citations import build_blocks, format_context
from rag.providers import (
    EmbeddingProviderError,
    HashingEmbeddings,
    LLMProviderError,
    OfflineExtractiveLLM,
    get_embeddings,
    get_llm,
)
from rag.providers.embeddings import _stem, tokenize


# --------------------------------------------------------------------------
# Offline embeddings
# --------------------------------------------------------------------------

def test_hashing_embeddings_are_deterministic_across_instances():
    """The whole offline story depends on this: separate processes and separate
    instances must produce identical vectors, or a persisted index is garbage."""
    a = HashingEmbeddings().embed_query("scaled dot-product attention")
    b = HashingEmbeddings().embed_query("scaled dot-product attention")
    assert a == b


def test_hashing_embeddings_are_l2_normalised():
    vec = HashingEmbeddings().embed_query("the Transformer encoder stack")
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)


def test_hashing_embeddings_respect_dimension():
    assert len(HashingEmbeddings(dim=256).embed_query("hello world")) == 256


def test_hashing_embeddings_reject_absurd_dimension():
    with pytest.raises(ValueError):
        HashingEmbeddings(dim=4)


def test_embed_documents_matches_embed_query():
    emb = HashingEmbeddings()
    assert emb.embed_documents(["chain of thought"])[0] == emb.embed_query("chain of thought")


def test_empty_text_gives_zero_vector_not_a_crash():
    vec = HashingEmbeddings().embed_query("")
    assert all(v == 0.0 for v in vec)


def test_related_text_scores_higher_than_unrelated():
    emb = HashingEmbeddings()
    query = emb.embed_query("How many layers does the Transformer encoder have?")

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    related = emb.embed_documents(["The Transformer encoder is a stack of six identical layers."])[0]
    unrelated = emb.embed_documents(["Sourdough bread requires a mature starter culture."])[0]
    assert cosine(query, related) > cosine(query, unrelated)


# --------------------------------------------------------------------------
# Tokenisation and stemming
# --------------------------------------------------------------------------

def test_stemmer_unifies_inflections():
    assert _stem("masking") == _stem("masked") == "mask"
    assert _stem("layers") == "layer"
    assert _stem("gradients") == "gradient"


def test_stemmer_does_not_shred_short_tokens():
    # Over-stemming collapses distinct technical terms; guard against it.
    assert _stem("is") == "is"
    assert _stem("bert") == "bert"
    assert _stem("512") == "512"


def test_tokenize_drops_stopwords_and_single_characters():
    tokens = tokenize("What is the value of d_model in the Transformer?")
    assert "the" not in tokens and "is" not in tokens
    assert "transformer" in tokens


# --------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------

def test_get_embeddings_offline():
    assert isinstance(get_embeddings("offline"), HashingEmbeddings)


def test_get_embeddings_rejects_unknown_provider():
    with pytest.raises(EmbeddingProviderError, match="Unknown embedding provider"):
        get_embeddings("word2vec")


def test_get_llm_offline():
    assert isinstance(get_llm("offline"), OfflineExtractiveLLM)


def test_get_llm_rejects_unknown_provider():
    with pytest.raises(LLMProviderError, match="Unknown LLM provider"):
        get_llm("gpt-9")


def test_anthropic_provider_fails_loudly_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
        get_llm("anthropic")


def test_batched_ollama_rejects_bad_batch_size(monkeypatch):
    """Construction must validate even though we never reach the server here."""
    pytest.importorskip("langchain_ollama")
    from rag.providers.embeddings import BatchedOllamaEmbeddings

    with pytest.raises(ValueError):
        BatchedOllamaEmbeddings(batch_size=0)


# --------------------------------------------------------------------------
# Offline reader behaviour
# --------------------------------------------------------------------------

def _ask(llm, question, docs):
    context = format_context(build_blocks(docs))
    system = config.SYSTEM_PROMPT.format(refusal=config.REFUSAL_TEXT, context=context)
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=question)]).content


def test_offline_reader_answers_and_cites(sample_docs):
    answer = _ask(OfflineExtractiveLLM(), "How many identical layers are in the encoder?", sample_docs)
    assert "[1]" in answer
    assert "N=6" in answer or "six" in answer.lower()


def test_offline_reader_refuses_when_context_is_irrelevant(sample_docs):
    answer = _ask(OfflineExtractiveLLM(), "What is the capital city of Australia?", sample_docs)
    assert answer == config.REFUSAL_TEXT


def test_offline_reader_refuses_with_no_context():
    llm = OfflineExtractiveLLM()
    answer = llm.invoke(
        [SystemMessage(content="no blocks here"), HumanMessage(content="anything?")]
    ).content
    assert answer == config.REFUSAL_TEXT


def test_offline_reader_is_deterministic(sample_docs):
    question = "How many parameters does BERT-LARGE have?"
    assert _ask(OfflineExtractiveLLM(), question, sample_docs) == _ask(
        OfflineExtractiveLLM(), question, sample_docs
    )


def test_offline_reader_never_cites_a_passage_it_was_not_given(sample_docs):
    from rag.citations import extract_citation_numbers

    answer = _ask(OfflineExtractiveLLM(), "How many layers are in the encoder stack?", sample_docs)
    assert extract_citation_numbers(answer) <= {1, 2}


def test_idf_weighting_downweights_ubiquitous_terms():
    llm = OfflineExtractiveLLM()
    blocks = ["training a model", "training another model", "training a third model"]
    weights = llm._token_weights({"train", "imagenet"}, blocks)
    # A term in every passage must weigh less than one in none of them.
    assert weights["train"] < weights["imagenet"]
