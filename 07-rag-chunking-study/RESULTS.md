# Results — Chunking Strategy Study

The deliverable requested by `README.md`: a comparison table across chunking strategies
plus a one-paragraph recommendation. Process and implementation notes live in `PLAN.md`;
this file is the outcome.

## Comparison table

| strategy (params) | # chunks | embed calls | Nova calls | ~$ to build | right-source hits (of 10) | answers correct (of 12) |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_300w_50ov | 175 | 175 | 0 | $0.0017 | 10/10 | 12/12 |
| **fixed_600w_100ov** | **91** | 91 | 0 | $0.0017 | 10/10 | **12/12** |
| fixed_800w_200ov | 73 | 73 | 0 | $0.0018 | 10/10 | 10/12 |
| separator | 890 | 890 | 0 | $0.0014 | 10/10 | 7/12 |
| sentence_3 | 802 | 802 | 0 | $0.0014 | 10/10 | 9/12 |
| sentence_5 | 484 | 484 | 0 | $0.0014 | 10/10 | 8/12 |
| sentence_8 | 306 | 306 | 0 | $0.0014 | 10/10 | 10/12 |
| semantic_llm | 196 | 196 | 28 | $0.0157 | 10/10 | 11/12 |

Right-source hit rate is out of 10, not 12 — the 2 unanswerable questions (grounding/
refusal check) have no expected source and are excluded from that column, per README's
definition. "Answers correct" is a keyword-match heuristic against 12 questions (8
single-fact, 2 cross-section, 2 unanswerable) — see `chunking-study/eval_questions.py` and
`chunking-study/eval_results.json` for the full per-question breakdown.

## Recommendation

**Right-source hit rate was 10/10 for every strategy** — on this corpus (short, clearly
headed markdown files, one topic per file) any reasonable chunking gets the retriever to
the correct *file*. That flattens the retrieval-quality signal README warned would happen
with raw cosine, but here it's the hit-rate metric itself that saturates — so the real
differentiator on this corpus is generation quality (answers correct), which tracks chunk
*size*, not strategy sophistication: bigger chunks carry more surrounding context for Nova
to work with, so **fixed_300w_50ov and fixed_600w_100ov both hit 12/12**, while the
fine-grained strategies (separator's median chunk is a single paragraph, sentence_3/5
similarly small) leave Nova reconstructing an answer from thinner, more fragmented context
and score visibly worse (separator 7/12, sentence_5 8/12). semantic_llm (11/12) is the
second-best strategy but costs ~9x more to build than any other option for a result
fixed_600w_100ov already beats.

**Ship fixed_600w_100ov (600 words, 100-word overlap).** It ties the best score (12/12)
at 91 chunks — a fifth the index size of separator/sentence for the same or better
accuracy — and costs $0.0017 to build, ~9x cheaper than semantic_llm for a strictly
better result on this corpus. fixed_300w_50ov ties it exactly and costs the same; 600w is
the better default going forward since it's cheaper per unit of *content* (91 vs 175
chunks covering the same corpus means less embedding-storage overhead at query time for
identical accuracy). Semantic/LLM-based chunking is the one to reach for only if the
corpus lacked clean structure (unlike this handbook) or size unpredictability actually
hurt naive chunking — neither is true here, so its ~9x build-cost premium isn't earned.

This is the config shipped: `homework/data/faiss_index_handbook/` (the default `rag_tool.py`
loads) holds the `fixed_600w_100ov` index — source-tagged, every answer citing its file.

## Possible next iteration (not done — optional per README)
- Try fixed_600w_100ov with a slightly larger overlap (150w) to see if the two remaining
  wrong answers close further.
- Optional: repeat the study on the `manager_playbook` corpus (README marks this optional).
