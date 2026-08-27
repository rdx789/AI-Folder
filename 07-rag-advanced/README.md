# Homework — Lesson 7 RAG Advanced · Assemble your production RAG pipeline

The lab measured the two advanced retrieval levers — metadata **filtering** and
**reranking** — one at a time. This homework puts them together with the **chunking
winner from Lesson 6** into **one complete, production-ready RAG pipeline**: tuned
chunking → metadata filtering → reranking → grounded, cited answers, with an eval
harness that *proves* every change instead of assuming it.

That pipeline **is** the retrieval core of your NovaOps **final project** — what your
assistant will retrieve with, and what every later retrieval improvement builds on. Do
this homework and you walk into the project with the hard part already built and
measured.

> Work in your **SDD** build (the filter+rerank mix) or by composing the lab's
> parts — either way, by Exercise 2 you want a single pipeline you can point an eval
> at. Each exercise's **Goal** line names the piece of the final project it produces.

## 1. Assemble the pipeline — finish the mix, wire in your Lesson 6 chunking winner

**Goal.** One `retrieve → filter → rerank → answer` pipeline, over an index chunked
the way your Lesson 6 study proved best — the assistant's retrieval core: the pipeline
your final-project assistant retrieves with. Everything below measures and tunes
*this*.

**What to build.**

- If you haven't finished the **SDD** build, complete it now: the filter+rerank MIX
  — access filter + subject planner + recency, then the Nova **listwise reranker**
  over a wide retrieve, kept to a small final top-k.
- Swap the lab's fixed **250/50-word** chunking (`CHUNK_WORDS` / `OVERLAP_WORDS` in
  `ingest.py`) for the **winning strategy + parameters** from your Lesson 6 chunking
  study, then re-ingest so filtering, reranking, and your chunking all run on one
  index.
- Carry Lesson 6's **source-citation** step into the answer — every claim names the
  file it came from. Production RAG answers are auditable — and a citation requirement
  is one your final-project assistant will have to meet.
- Confirm end-to-end: an employee query is access-filtered, the planner narrows by
  subject, reranking curates the top-k, and the answer cites its sources.

**Prompt hint.**
> "Assemble my filter+rerank pipeline into one `retrieve → filter → rerank → answer`
> flow. In ingest, replace the fixed 250/50-word chunking with <my Lesson 6 winner:
> strategy + params> and re-index. In the answer step, tag each context chunk with
> its `source` and require every claim to cite its file. Run one employee question
> and one manager question end-to-end and show the access filter, the planned
> subjects, the reranked top-k, and the cited answer."

## 2. Prove it works — a labeled test set + eval harness, then improve

**Goal.** Measure **retrieval** and **generation** so you tune with evidence, not
vibes — your final-project eval harness (LLM judges for the answer + labeled
Recall@k / MRR for retrieval). This is the instrument you run every later change
through.

**What to build.**

- Grow `data/eval_questions.jsonl` into a real **test set**: start from the fixed
  question set you wrote in Lesson 6, add an `expected_source` (the file that truly
  answers each answerable question), and cover single-fact, cross-source, access
  (expect refusal), and unanswerable cases.
- Write an eval that scores each answer with the three **LLM judges** (faithfulness /
  context-relevance / completeness) **and** computes **Recall@k** and **MRR** against
  `expected_source`. **Baseline** your Exercise-1 pipeline.
- Then **improve and re-measure**: tune chunk config, retrieve-N, rerank top-k,
  planner breadth — one change at a time — and keep what moves the numbers. Report
  before/after and name which lever paid off.

**Prompt hint.**
> "Extend `data/eval_questions.jsonl` with an `expected_source` per answerable row
> and more questions across single-fact / cross-source / access / unanswerable.
> Write `eval_harness.py` that runs my Exercise-1 pipeline over the set and prints,
> per config, the three LLM judge scores AND Recall@k + MRR (by checking each
> retrieved chunk's `source`). Baseline the pipeline, then let me tweak chunking /
> retrieve-N / rerank top-k and re-run to compare. Where do the labeled retrieval
> metrics and the judges' context-relevance agree or disagree?"

## 3. Tuning lever — hybrid search (BM25 + vector)

**Goal.** A retrieval-tuning mode for your assistant: the fix for exact-term queries
(an error code, a policy name) that pure vectors fumble — validated against your
Exercise-2 harness, not assumed.

**What to build.** Run the query as **both** a BM25 `match` on the `text` field and
the k-NN query, each top-N. BM25 scores and cosine-ish scores aren't comparable, so
fuse the two lists with **reciprocal-rank fusion** (rank-based — no normalization
guesswork). Keep the `audience` filter on both. Then point your Exercise-2 harness at
the hybrid retriever and see which query types improve.

**Prompt hint.**
> "Add `hybrid_search(client, query, audience, top_k)` that runs a BM25 `match` on
> `text` and the existing k-NN, each top-20, and fuses them with reciprocal-rank
> fusion. Keep the access filter on both. Point my Exercise-2 harness at it and
> compare Recall@k / faithfulness / context-relevance against the vector-only
> baseline — which question types improve most?"

## 4. Tuning lever — query rewriting

**Goal.** A retrieval-tuning mode for your assistant: turn a vague or multi-part user
message into a focused search query *before* retrieval — again measured against your
harness.

**What to build.** A `rewrite_query` step (one Nova call) that turns a conversational
question ("what happens to my time off if I quit?") into a focused retrieval query
("unused vacation payout on resignation / termination"), or splits a two-part
question into two. Retrieve with the rewritten query, and measure whether context
relevance improves — especially on the cross-source case.

**Prompt hint.**
> "Add `rewrite_query(question)` (one Nova call) that rewrites a conversational
> question into a focused search query and, for a two-part question, returns two
> queries. Wire it in before retrieval, retrieve per rewritten query and merge, and
> compare context-relevance and completeness on the cross-source case with rewriting
> on vs. off using my Exercise-2 harness."

---

*All four exercises are on-project — no optional off-project work this lesson. Finish
them and you hold a **complete, production-ready RAG pipeline**: tuned chunking,
access-controlled retrieval, reranking, source citations, and an eval harness that
keeps it honest — the retrieval layer your final-project assistant ships.*
