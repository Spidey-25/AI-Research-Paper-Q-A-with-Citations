# Evaluation results

Measured run, not an estimate. Reproduce with:

```bash
python scripts/fetch_corpus.py
python -m rag.ingest --rebuild
python eval/run_eval.py
```

| Setting | Value |
|---|---|
| Test set | `eval/testset.json` - 20 hand-written questions (15 in-corpus, 5 out-of-corpus) |
| Corpus | 6 arXiv papers, 19 pages, 46 chunks (`corpus/manifest.json`) |
| Embedding provider | `offline` (deterministic hashed bag-of-words) |
| LLM provider | `offline` (deterministic extractive reader) |
| Chunking | 1200 characters, 200 overlap |
| Retrieval | top-k = 5, cosine, relevance floor 0.0 |
| Refusal threshold | 0.40 IDF-weighted coverage |
| Date | 2026-08-10 |

---

## Headline numbers

| Metric | Score | Detail |
|---|---|---|
| **Retrieval accuracy** | **100.0%** | 15/15 in-corpus questions retrieved the correct paper in the top 5 |
| **Answer correctness** | **60.0%** | 9/15 answers contained every required fact |
| **Citation rate** | **86.7%** | 13/15 answers carried a valid inline citation |
| **Refusal accuracy** | **100.0%** | 5/5 out-of-corpus questions correctly refused |
| Hallucinated answers | **0** | no out-of-corpus question was answered |
| False refusals | 2 | Q13, Q14 wrongly refused |
| Fabricated citations | **0** | no `[n]` marker pointed outside the retrieved set |
| Mean latency | 0.008 s | offline providers, no network |

**Read these numbers correctly.** They were produced with the *offline* providers,
which have no semantic model behind them: retrieval is lexical bag-of-words and
generation is sentence extraction, not language generation. So:

* Retrieval accuracy, citation rate, refusal accuracy and fabricated-citation
  count measure **the plumbing** - metadata propagation, the citation contract,
  the refusal gate, the harness itself. These are meaningful.
* **Answer correctness of 60% is a floor, not a quality claim.** A reader that
  can only copy sentences will miss facts that a real model would paraphrase
  correctly. Running the same harness with `RAG_LLM_PROVIDER=anthropic` and
  `RAG_EMBEDDING_PROVIDER=ollama` is expected to move this number substantially;
  that run has not been performed here (see *What has not been measured*).

---

## Per-question results

`RETR` retrieved the right paper · `ANS` contained every required fact ·
`CITE` valid inline citation · `REFUSE` refused/answered as it should have

```
ID    TYPE            RETR   ANS   CITE  REFUSE   QUESTION
----------------------------------------------------------
Q01   in-corpus        Y      Y     Y       Y     How many identical layers are in the Transformer's enco...
Q02   in-corpus        Y      N     Y       Y     Why does scaled dot-product attention divide the dot pr...
Q03   in-corpus        Y      N     Y       Y     What is the value of d_model in the Transformer, and wh...
Q04   in-corpus        Y      Y     Y       Y     What BLEU score did the Transformer achieve on the WMT ...
Q05   in-corpus        Y      N     Y       Y     What percentage of WordPiece tokens does BERT mask duri...
Q06   in-corpus        Y      Y     Y       Y     When BERT selects a token for masking, what does it rep...
Q07   in-corpus        Y      Y     Y       Y     How many parameters do BERT-BASE and BERT-LARGE have?
Q08   in-corpus        Y      N     Y       Y     Which corpora were used to pre-train BERT and how large...
Q09   in-corpus        Y      Y     Y       Y     How many parameters does GPT-3 have and how many layers...
Q10   in-corpus        Y      Y     Y       Y     In the GPT-3 paper, how many examples K are typically u...
Q11   in-corpus        Y      Y     Y       Y     What is the difference between the RAG-Sequence and RAG...
Q12   in-corpus        Y      Y     Y       Y     Which retriever architecture does the RAG paper use, an...
Q13   in-corpus        Y      N     N       N     Which components does the RAG paper keep fixed during t...
Q14   in-corpus        Y      N     N       N     How are the LoRA matrices A and B initialised, and what...
Q15   in-corpus        Y      Y     Y       Y     How many chain-of-thought exemplars were used to prompt...
Q16   out-of-corpus    -      Y     Y       Y     What is the capital city of Australia?
Q17   out-of-corpus    -      Y     Y       Y     What revenue did Woolworths Group report in FY2024?
Q18   out-of-corpus    -      Y     Y       Y     What learning rate schedule does the Adam optimiser use...
Q19   out-of-corpus    -      Y     Y       Y     Which team won the 2023 Cricket World Cup final?
Q20   out-of-corpus    -      Y     Y       Y     What is the current cash rate set by the Reserve Bank o...
```

---

## Failure analysis

Every failure below is a limitation of the offline extractive reader, not of
retrieval: **retrieval was correct on all 15 in-corpus questions.**

| ID | Failure | Cause |
|---|---|---|
| Q02 | Missing "gradients" | Reader quoted the two sentences with the highest term overlap; the sentence explaining *why* scaling is needed ranked third. It cannot summarise across sentences. |
| Q03 | Missing "512" | Same: it selected the sentences naming `sub-layers` over the one carrying the number. |
| Q05 | Missing "15%" | "WordPiece" appears in both the vocabulary sentence and the masking sentence; the vocabulary sentence scored higher on overlap. |
| Q08 | Missing corpus names | Question says "corpora"; the paper says "corpus"/"BooksCorpus". The stemmer deliberately does not handle irregular Latin plurals, so the query and the passage never lexically meet. A semantic embedding model would not have this problem. |
| Q13, Q14 | Refused a valid question | IDF-weighted coverage fell below the 0.40 threshold (0.387 and 0.249). These are the cost of setting the threshold conservatively - see below. |

Q08 is the single clearest demonstration of why this project keeps a real
embedding provider: **a bag-of-words retriever cannot know that "corpora" and
"corpus" are the same word**, and no amount of prompt engineering fixes that.

---

## The refusal threshold trade-off

The offline reader refuses when the IDF-weighted share of the question's content
words found in a passage falls below a threshold. That threshold trades false
refusals against hallucinations. Measured across the full test set:

| Threshold | Retrieval | Answer correctness | Citation rate | Refusal accuracy | Hallucinations | False refusals |
|---|---|---|---|---|---|---|
| 0.20 | 100.0% | 66.7% | 100.0% | 80.0% | **1** | 0 |
| 0.30 | 100.0% | 60.0% | 93.3% | 80.0% | **1** | 1 |
| 0.35 | 100.0% | 60.0% | 93.3% | 80.0% | **1** | 1 |
| **0.40** | **100.0%** | **60.0%** | **86.7%** | **100.0%** | **0** | **2** |
| 0.45 | 100.0% | 53.3% | 80.0% | 100.0% | 0 | 3 |
| 0.50 | 100.0% | 53.3% | 80.0% | 100.0% | 0 | 3 |

**0.40 is shipped as the default.** The reasoning is a product judgement rather
than a metric-maximising one: a false refusal is a visible nuisance the user can
work around by rephrasing, whereas a confident wrong answer is a silent failure
the user cannot detect. For a system whose selling point is trustworthy
citations, refusing too often is the correct direction to be wrong in.

**Disclosure:** this threshold is the one hyperparameter fitted against the
evaluation set. With only 20 questions there is no held-out split, so the
refusal accuracy of 100% should be read as optimistic. A larger test set with a
proper train/test split is the first thing this harness needs.

### Why Q18 is in the test set

Q18 - *"What learning rate schedule does the Adam optimiser use to train a
convolutional network on ImageNet?"* - is the deliberately adversarial case. It
is dense with vocabulary the corpus genuinely contains (`Adam`, `learning rate`,
`training`, `convolutional`, `network`) while asking about something no paper in
the corpus covers.

An early version of the reader answered it, quoting LoRA's Adam sentence, because
plain unweighted term overlap reached 0.50 on that passage. The fix was to weight
each question term by its inverse document frequency across the retrieved
passages: terms appearing in every passage (`training`, `model`) carry little
weight, while terms appearing in none (`ImageNet`, `optimiser`, `schedule`) are
exactly the evidence that the question is unanswerable. That change alone took
hallucinations from 1 to 0.

**This is the concrete case for building the harness.** The hallucination was
invisible from reading output - the answer was fluent, correctly cited, and drawn
from a real passage. Only a test set with known-unanswerable questions surfaced it.

---

## What the harness measures

| Metric | Definition | Why it is separate |
|---|---|---|
| Retrieval accuracy | Did every paper listed in `expected_sources` appear in the top-k? | Caps everything downstream. If retrieval misses, no model can recover. |
| Answer correctness | Does the answer contain at least one accepted surface form from every required keyword group? | Deterministic and free. No LLM judge, so the score cannot drift between runs. |
| Citation rate | At least one inline `[n]`, and every `[n]` within the retrieved range. | A citation to `[9]` when 5 passages were retrieved is a fabricated citation, which is worse than none. |
| Refusal accuracy | On out-of-corpus questions only: was the refusal string emitted? | This is the anti-hallucination number. |

Keyword containment is a blunt instrument: it can reward an answer that contains
the right token in the wrong sentence, and punish a correct paraphrase. It was
chosen over an LLM judge because it is reproducible, costs nothing, and needs no
credentials - the same reasons the offline providers exist. A semantic judge is
listed under future improvements in the README.

---

## What has *not* been measured

Stated plainly, because the point of this document is to be trustworthy:

1. **No production-provider run.** Every number here comes from the offline
   providers. The system has not been scored with `embeddinggemma` embeddings
   and a Claude model, because the machine this was built on has neither an
   Ollama server nor an API key. The harness is provider-agnostic and the run is
   two environment variables away, but until it is executed there is **no
   measured claim about production answer quality**.
2. **No held-out split.** 20 questions, one of which (the refusal threshold) was
   fitted on all 20.
3. **No retrieval ablation.** k, chunk size and overlap were set once by
   judgement, not swept.
4. **Single corpus, single domain.** Six ML papers with a shared vocabulary.

### Reproducibility note on the corpus

These numbers were produced on a build machine whose network policy blocks
`arxiv.org`, so `scripts/fetch_corpus.py` could not run there. To get an
end-to-end measurement, the six papers' text was rendered into local PDFs of the
same content and indexed instead.

The consequences, stated exactly:

* The test-set questions target facts stated in the papers' abstracts and main
  sections, so the **questions and expected answers hold against the genuine
  arXiv PDFs**.
* **Page numbers and retrieval scores will differ** when the real PDFs are used,
  since the real papers are longer and laid out differently. `expected_sources`
  is matched on arXiv id, not page, precisely so the test set survives this.
* Anyone reproducing this should run `python scripts/fetch_corpus.py` against
  the real arXiv and **regenerate this table**. Do not treat the numbers above
  as verified against the published PDFs.
