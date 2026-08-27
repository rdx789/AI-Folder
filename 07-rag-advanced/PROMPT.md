# PROMPT — RAG pipeline

**Overview.** One complete, production-ready RAG pipeline: tuned chunking →
metadata filtering → reranking → grounded, cited answers, backed by an eval
harness that proves every change instead of assuming it.

**Environment.**

- Location: provided plumbing (`client.py`, `judges.py`) and all pipeline scripts
  live in `code/`.
- Data: source docs under `data/handbook/` (16 files) and
  `data/manager_playbook/` (14 files). Each file has YAML frontmatter —
  `last_updated`, `corpus` (`handbook` | `manager_playbook`), `audience`
  (`all` | `manager`) — read directly from frontmatter, not inferred from folder.
- Credentials: `.env` in `homework/` (gitignored) provides AWS/Bedrock +
  OpenSearch Serverless creds; `OPENSEARCH_COLLECTION` is the live collection
  name, resolved to an endpoint by `code/client.py`.

---

## 1. Assemble the pipeline — status: done

**Goal.** One `retrieve → filter → rerank → answer` pipeline over a single
chunked index.

**What to build.**

- The filter+rerank MIX: access filter + subject planner + recency, then the
  Nova **listwise reranker** over a wide retrieve, kept to a small final top-k.
- Fixed **600w / 100ov** chunking (`CHUNK_WORDS` / `OVERLAP_WORDS` in
  `ingest.py`). Filtering, reranking, and chunking all run on one index.
- Index mapping includes a `key` field — `"key": {"type": "keyword"}` — the
  corpus/filename identifying the source doc a chunk came from.
- A source-citation step in the answer — every claim names the file it came
  from, so answers stay auditable.
- Confirm end-to-end: an employee query is access-filtered, the planner narrows
  by subject, reranking curates the top-k, and the answer cites its sources.

**Built as:** `code/create_index.py` (index + mapping, `--force` to
drop/recreate — no-ops otherwise so re-runs never wipe a populated index).
Vector field: HNSW / **faiss** engine / **innerproduct** space — `nmslib` is
deprecated and unreliable on OpenSearch Serverless, and Titan embeds with
`normalize:true`, so inner product == cosine. `MAPPING` is defined once here
and imported by `ingest.py`.
`code/subjects.py` (14-label subject vocabulary + Nova tagger, shared by
ingest and the planner), `code/ingest.py` (parses frontmatter, chunks,
tags, embeds, bulk-indexes — chunks cheaply first and re-ingests only if the
chunk count changed; `--force` drops and rebuilds unconditionally),
`code/pipeline.py` (`retrieve` N=20 wide k-NN → `filter_by_subject_and_recency`
→ `rerank` to K=4 → `answer`; the no-context refusal string is a single
`REFUSAL` constant the answer prompt is pinned to). Verified on one employee
and one manager query, showing the access filter, planned subjects, reranked
top-k, and cited answer.

**Prompt hint.**
> "Assemble my filter+rerank pipeline into one `retrieve → filter → rerank →
> answer` flow. In ingest, use 600w-100ov chunking to build the index. In the
> answer step, tag each context chunk with its `source` and require every claim
> to cite its file. Run one employee question and one manager question
> end-to-end and show the access filter, the planned subjects, the reranked
> top-k, and the cited answer."

---

## 2. Prove it works — a labeled test set + eval harness, then improve — status: done

**Goal.** Measure retrieval and generation so tuning is evidence-based — LLM
judges for the answer, labeled Recall@k / MRR for retrieval. This is the
instrument every later change is run through.

**What to build.**

- Grow `data/eval_questions.jsonl` into a real test set: an `expected_source`
  per answerable question (the file that truly answers it), covering
  single-fact, cross-source, access (expect refusal), and unanswerable cases.
- An eval that scores each answer with the three LLM judges (faithfulness /
  context-relevance / completeness) and computes Recall@k and MRR against
  `expected_source`. Baseline the pipeline.
- Improve and re-measure: tune chunk config, retrieve-N, rerank top-k, planner
  breadth — one change at a time — and keep what moves the numbers. Report
  before/after and name which lever paid off.

**Built as:** `data/eval_questions.jsonl` extended to 17 questions across
single-fact / cross-source / access / unanswerable / exact-term, each with
`expected_source` (`[]` for refusal cases).
`code/eval_harness.py` runs the pipeline per question and reports Recall@k /
MRR (checking whether the reranked top-k's chunk `key`s contain
`expected_source`) plus the three judges for answerable questions, and refusal
accuracy for access/unanswerable ones.

**Baseline (N=20, k=4, vector-only, 17 questions, clean 165-chunk index):**
Recall@k 1.00, MRR 0.96, faithfulness 0.98, context-relevance 1.00,
completeness ~0.90, refusal accuracy 1.00 (4/4). Recall@k and context-relevance
are perfect and stable — the wide retrieve + subject filter always deliver the
right doc and nothing off-topic. The residual gaps are small: MRR 0.96 is two
rank-2 placements (`XSRC_ONBOARD`, `SPEND_THRESHOLD`) the reranker doesn't
fully fix; completeness sits at 0.89–0.93 across runs (judge nondeterminism on
borderline facts like `K401_MATCH`, whose answer under-states one required
fact). 600w/100ov chunking holds up — no lever tried (hybrid search,
retrieve-N, rerank top-k) beat the vector-only default.

Three issues surfaced and fixed while baselining:
- Two "access" test questions were answerable from `audience: all` docs (not a
  real access-control test — retargeted at content that exists only in
  `manager_playbook/`).
- The model's natural refusal phrasing didn't match `judges.refused()`'s
  keyword heuristic — fixed by pinning the pipeline's refusal wording to one
  exact sentence.
- `ingest.py --force` appended instead of replacing (AOSS Serverless assigns
  its own `_id`s and has no `_delete_by_query`), so a re-run left two copies of
  every chunk; combined with AOSS's async, occasionally-lossy single writes,
  an early run was missing a `benefits-and-perks.md` chunk and tanked
  completeness on `PTO_ROLLOVER` / `SABBATICAL`. Fixed: `--force` now drops and
  recreates the index, ingest uses the bulk helper (which raises on any
  rejected doc instead of silently short-counting), then polls `count()` and
  prints `Bulk-indexed N/N ... (M searchable)`.

**Prompt hint.**
> "Extend `data/eval_questions.jsonl` with an `expected_source` per answerable
> row and more questions across single-fact / cross-source / access /
> unanswerable. Write `eval_harness.py` that runs the pipeline over the set and
> prints, per config, the three LLM judge scores AND Recall@k + MRR (by
> checking each retrieved chunk's `source`). Baseline the pipeline, then let me
> tweak chunking / retrieve-N / rerank top-k and re-run to compare. Where do
> the labeled retrieval metrics and the judges' context-relevance agree or
> disagree?"

---

## 3. Tuning lever — hybrid search (BM25 + vector) — status: done

**Goal.** A retrieval-tuning mode for the assistant: the fix for exact-term
queries (an error code, a policy name) that pure vectors fumble — validated
against the harness, not assumed.

**What to build.** Run the query as both a BM25 `match` on the `text` field
and the k-NN query, each top-N. BM25 scores and cosine-ish scores aren't
comparable, so fuse the two lists with reciprocal-rank fusion (rank-based — no
normalization guesswork). Keep the `audience` filter on both. Point the
harness at the hybrid retriever and see which query types improve.

**Built as:** `code/hybrid_search.py` — `hybrid_search(client, query,
audience, top_k, top_n=20)` runs BM25 + k-NN with the audience filter on both
and fuses by RRF (k=60). Wired into `pipeline.run(..., use_hybrid=True)`.

**Result — hybrid did not help on this eval set** (N=20, k=4, 17 questions,
clean 165-chunk index):

| metric | vector-only | hybrid (BM25+kNN, RRF) |
| --- | --- | --- |
| Recall@k | 1.00 | 1.00 |
| MRR | 0.96 | 0.92 |
| faithfulness | 0.98 | 0.92 |
| context-relevance | 1.00 | 1.00 |
| completeness | 0.93 | 0.82 |

(Vector and hybrid measured in the same session on the same index; completeness
carries ~±0.04 judge noise run-to-run, the other five metrics are stable.)

Hybrid matched vector on Recall@k and context-relevance but regressed MRR,
faithfulness, and completeness. The clearest loss: `MOON_PT` ("part-time job
at another software company") completeness collapsed to 0.00 — BM25 matched
the literal phrase against off-topic chunks and pushed `moonlighting.md`'s
answer content out of the reranked top-k.

**The exact-term test didn't rescue it either.** `SPEND_THRESHOLD` — a
deliberate literal lookup ("USD 1,500", "Tom Baker", `handbook/titles-for-ops.md`)
added to the set for exactly this — scored *identically* under both retrievers
(Recall@k 1, MRR 0.50, all judges 1.00): the reranker already recovers the
right chunk from vector retrieval, so BM25 adds nothing on top. On a corpus
this small (165 chunks) with a strong listwise reranker, hybrid's upside is
marginal and its lexical noise is real. Kept as opt-in (`use_hybrid=True`),
not the default.

**Prompt hint.**
> "Add `hybrid_search(client, query, audience, top_k)` that runs a BM25
> `match` on `text` and the existing k-NN, each top-20, and fuses them with
> reciprocal-rank fusion. Keep the access filter on both. Point the harness at
> it and compare Recall@k / faithfulness / context-relevance against the
> vector-only baseline — which question types improve most?"

---

## Tooling — the tuning loop

`.claude/skills/rag-eval-loop/SKILL.md` packages the Exercise-2 improve loop as
a skill: baseline → change **one** lever → re-ingest only if chunk config
changed → re-measure with `eval_harness` → decide keep/revert → record
before/after in this file. It encodes the conventions that are easy to get
wrong — one change at a time, don't `--force` re-ingest unless chunking
actually changed (real Bedrock spend), a regressing lever stays wired but
opt-in, results tables live in §2 / §3. Invoke it whenever tuning or comparing
configs.

## AOSS Serverless constraints (learned the hard way)

The collection is OpenSearch **Serverless**, which drops several APIs the
scripts would otherwise use: no `_delete_by_query`, no explicit document
`_id`s, no `_refresh`, and individual writes are async and occasionally lossy.
Consequences baked into the code: `ingest.py --force` drops+recreates the
index (only way to clear it), uses `helpers.bulk` (which raises on rejected
docs rather than silently short-counting), and polls `count()` afterward
because the searchable count lags the write.
