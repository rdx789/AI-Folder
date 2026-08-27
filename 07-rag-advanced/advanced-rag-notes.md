# Advanced RAG — working notes

A summary of the advanced-retrieval topic, written against the NovaOps
pipeline in this folder. Covers the pipeline shape, each tuning lever, how
it's measured, and the vector-store stack that sits underneath.

---

## 1. The pipeline

One flow, one index:

```
retrieve  →  filter  →  rerank  →  answer
 (wide)      (cheap)     (dear)     (grounded + cited)
```

| Stage | What it does | Cost | Notes |
|---|---|---|---|
| **retrieve** | wide k-NN over the index, `audience` filter applied at query level | 1 embed call + 1 search | N=20 candidates. Wide on purpose — recall now, precision later. |
| **filter** | subject planner narrows to on-topic chunks; recency sort breaks ties | 1 Nova call (planner) | *Soft* narrowing: if the subject filter would empty the set, fall back to the full 20. |
| **rerank** | Nova listwise reranker scores all candidates together, keeps top-k | 1 Nova call | Listwise (sees all chunks at once) beats pointwise (scores each alone) — it can compare candidates. K=4. |
| **answer** | Nova answers from the top-k only, every claim cites its source file | 1 Nova call | Refusal wording is pinned to one exact sentence so the eval's keyword check is reliable. |

**Why wide-then-narrow:** retrieval (bi-encoder cosine) is cheap but coarse;
reranking (cross-encoder-style, the whole query+chunk in one context) is a
stronger relevance signal but too expensive to run over the whole corpus. So
retrieve wide with the cheap signal, then spend the expensive signal only on
the survivors.

---

## 2. Chunking

- **600 words / 100 overlap**, fixed. The overlap keeps a fact that straddles
  a boundary recoverable from at least one chunk.
- Frontmatter (`last_updated`, `corpus`, `audience`) is read from the file and
  attached to every chunk as metadata — not inferred from the folder.
- **600w/100ov held up.** An early run showed low completeness on
  `benefits-and-perks.md` questions (`PTO_ROLLOVER`, `SABBATICAL` at 0.00) and
  it looked like a chunking problem — big single-topic file, few chunks, fact
  split from its chunk. It wasn't: the real cause was an **index-integrity
  bug** (duplicate-append + a dropped write — see §5). On a clean index those
  questions score
  0.83–0.97 and no chunk-config change beat 600/100. Lesson: rule out the data
  layer before blaming the algorithm.

---

## 3. Filtering (metadata)

Three filters, different mechanisms:

| Filter | Mechanism | Hard or soft |
|---|---|---|
| **access** (`audience`) | `terms` filter inside the k-NN query — employee sees `all`, manager sees `all` + `manager` | **hard** — a security boundary, never bypassed |
| **subject** | planner (Nova, forced tool call → 1–3 labels from a closed 14-label vocabulary) tags the query; keep chunks whose tags intersect | **soft** — falls back to unfiltered if it would empty the set |
| **recency** | sort by `last_updated` descending | tie-breaker only |

The subject vocabulary is a **closed list** shared by the ingest tagger and
the query planner, so both sides always speak the same labels. Structured
output (a forced `submit_subjects` tool call) beats parsing prose.

---

## 4. Reranking

- **Listwise**, one Nova call: the query + all candidate chunks go in
  together, the model returns them ranked by relevance via a forced
  `submit_ranking` tool call.
- Stronger signal than the retrieval embedding because the model reads the
  full query and full chunk in one context, rather than compressing each to a
  vector and comparing geometry.
- Keeps the final set small (k=4) so the answer step sees little noise.
- On a tiny corpus with a strong reranker, upstream retrieval quality matters
  less than usual — the reranker recovers a lot. (This is why hybrid search
  didn't help here; see §6.)

---

## 5. Evaluation — two instruments

You cannot tune what you cannot measure. Every change in this project was run
through both:

### Labeled retrieval metrics
Against `expected_source` (the file that truly answers each question):

- **Recall@k** — was *any* expected source in the reranked top-k? (0/1)
- **MRR** — reciprocal rank of the first correct hit (1.0 at rank 1, 0.5 at
  rank 2, …). Catches "right doc, wrong position".

### LLM-as-judge (no golden answers needed)
Each is one Nova call with a rubric in the system prompt and a forced
`submit_score` tool call:

- **faithfulness** — is every claim in the answer supported by the context?
  (grounding — catches hallucination)
- **context-relevance** — what fraction of retrieved chunks are on-topic?
  (retrieval precision — an off-topic chunk lowers it even if other chunks
  already answer the question)
- **completeness** — what fraction of the question's required facts made it
  into the answer? (catches a fact buried by too-large k, or dropped by
  too-small k)

Plus a **refusal check** (keyword heuristic, not an LLM call) for the
access / unanswerable cases — a correct pipeline must *decline* these.

### Where the two agree / disagree
- **Agree:** when context-relevance is low, completeness usually follows —
  noisy context crowds out the needed fact.
- **Disagree:** Recall@k can be 1.0 while completeness is 0.0 — the right
  *document* was retrieved, but the specific *chunk* with the fact wasn't in
  the top-k. Recall@k is a document-level signal; completeness is fact-level.
  (This is exactly the shape the §5 index bug produced — a good diagnostic for
  "retrieval is fine, something downstream isn't".)

### This pipeline's numbers (N=20, k=4, vector-only, 17 questions, clean index)
| metric | score |
|---|---|
| Recall@k | 1.00 |
| MRR | 0.96 |
| faithfulness | 0.98 |
| context-relevance | 1.00 |
| completeness | 0.89–0.93 (judge noise) |
| refusal accuracy | 1.00 (4/4) |

### Index-integrity bug (worth remembering)
AOSS Serverless is constrained vs. classic OpenSearch: **no explicit `_id`s**,
**no `_delete_by_query`**, **no `_refresh`**, and indexing is **asynchronous
and can silently drop an individual write** under load. Consequences here:
- `client.index()` returning 200 doesn't mean the doc is searchable yet — the
  count lags for seconds.
- `ingest.py --force` originally *appended* (server-assigned `_id`s), so a
  second run left two copies of every chunk.

Fix: `--force` **drops and recreates** the index (the only way to clear it),
then **bulk-indexes** via `opensearchpy.helpers.bulk`, which raises
`BulkIndexError` listing any rejected docs instead of silently short-counting.
Then poll `count()`.

Symptom of the original bug was completeness 0.00 on two questions whose
Recall@k was 1.0 — the "retrieval fine, answer broken" pattern. Always confirm
`chunks sent == chunks searchable` before trusting an eval number.

---

## 6. Tuning lever — hybrid search (BM25 + vector)

**Idea:** pure vectors fumble exact-term queries (an error code, a policy
name, a dollar figure) because a paraphrase-tolerant embedding dilutes literal
term overlap. Run the query as **both** a BM25 `match` and a k-NN search, each
top-N, keep the `audience` filter on both, and fuse.

**Fusion — Reciprocal Rank Fusion (RRF):** BM25 scores and cosine scores live
on different, incomparable scales, so never mix raw scores. RRF looks only at
*rank position* in each list:

```
score(doc) = Σ  1 / (k + rank_in_list)      k = 60 (standard damping constant)
           lists containing doc
```

No normalization guesswork; a doc that ranks well in *either* list floats up.

**Result on this project — hybrid did not help** (clean 165-chunk index):
| metric | vector-only | hybrid |
|---|---|---|
| Recall@k | 1.00 | 1.00 |
| MRR | 0.96 | 0.92 |
| faithfulness | 0.98 | 0.92 |
| context-relevance | 1.00 | 1.00 |
| completeness | 0.93 | 0.82 |

- The eval questions are natural-language paraphrases, not literal lookups, so
  BM25 mostly injected lexically-similar-but-off-topic chunks that crowded out
  better vector hits before the reranker saw them. `MOON_PT` ("part-time job at
  another software company") lost its answer content this way — completeness
  0.00 under hybrid vs 1.00 under vector.
- Adding a deliberate exact-term question (`SPEND_THRESHOLD` — "USD 1,500",
  "Tom Baker") did **not** rescue hybrid: it scored *identically* under both
  retrievers, because the listwise reranker already recovers the right chunk
  from vector retrieval alone.
- **Takeaway:** hybrid earns its keep on large corpora with genuine
  exact-term traffic and a weaker/absent reranker. On a small corpus behind a
  strong reranker, its lexical noise outweighs its upside. Kept as opt-in
  (`use_hybrid=True`), not the default.

---

## 7. Tuning lever — query rewriting (not yet built)

**Idea:** one Nova call *before* retrieval that turns a vague or multi-part
user message into a focused search query — "what happens to my time off if I
quit?" → "unused vacation payout on resignation / termination" — or splits a
two-part question into two queries, retrieve per query, merge.

Expected to help most on the **cross-source** case, where one embedding of a
compound question sits between two topics and retrieves neither well. Measure
context-relevance and completeness on those questions with rewriting on vs.
off.

---

## 8. The vector-store stack

"Which vector DB" and "which algorithm" are different layers. Top to bottom:

| Layer | What it is | This project | Alternatives |
|---|---|---|---|
| **Vector database / service** | stores vectors + metadata, runs queries, does filtering, scaling, persistence | **AOSS** (Amazon OpenSearch Serverless) | Pinecone, Weaviate, Qdrant, pgvector, a local FAISS file |
| **Engine** | the library that builds the ANN index inside that DB | **faiss** | nmslib *(deprecated on Serverless)*, lucene |
| **Algorithm / index type** | the data structure used for approximate search | **HNSW** | IVF, flat / brute-force, PQ |
| **Distance metric** | how "close" is scored | **inner product** (= cosine, because Titan embeds with `normalize: true` → unit vectors) | cosine, L2 |

Key points:

- **HNSW is not a peer of a vector DB** — it's a knob *inside* one. "Vector DB"
  = the top row; HNSW = the third row.
- **HNSW** = fast, high recall, more memory. **Brute-force** = exact, fine up
  to ~10k–100k vectors. This corpus is **165 chunks** — brute-force would give
  identical results, HNSW is harmless overkill and the default, so left as-is.
- **faiss over nmslib**: nmslib is deprecated and not reliably supported on
  OpenSearch Serverless — the original `nmslib` mapping is why nothing ran
  until it was switched.
- **inner product over cosine**: identical math on normalized vectors, one
  fewer normalization per query. Use `cosinesimil` only if you want the
  mapping to read more explicitly.

### Store choice for the final project: stay on AOSS
- Hybrid search needs BM25 — OpenSearch has it built in; a raw FAISS library
  would mean hand-rolling BM25 separately.
- Metadata filtering (`audience`, `subjects`, recency) is native to the k-NN
  query in OpenSearch; manual in a FAISS-file setup.
- The NextGen Serverless collection scales to zero when idle, so the standing
  cost argument for going local is largely moot.
- Only switch to a local FAISS index if a zero-cloud-dependency final project
  is an explicit goal.

---

## 9. One-line recap of each lever

| Lever | When it pays off |
|---|---|
| Wider retrieve-N | recall is the bottleneck (expected doc not in candidate set) |
| Smaller chunks | completeness low despite Recall@k = 1 (fact split from its chunk) |
| Bigger rerank k | a needed fact keeps landing just outside the top-k |
| Narrower planner | context-relevance low from off-topic-but-adjacent subjects |
| Hybrid (BM25+RRF) | large corpus, real exact-term traffic, weak/no reranker |
| Query rewriting | vague or multi-part questions, especially cross-source |
| Recency sort | corpus has stale + fresh versions of the same policy |
