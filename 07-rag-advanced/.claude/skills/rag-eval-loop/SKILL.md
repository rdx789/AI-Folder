---
name: rag-eval-loop
description: Run one evidence-based tuning iteration on the NovaOps RAG pipeline — baseline, change ONE lever, re-measure with eval_harness, decide keep/revert, and record before/after in PROMPT.md. Use when asked to tune, improve, or measure retrieval/answer quality, try a lever (chunking, retrieve-N, rerank top-k, planner, hybrid, query rewrite), or compare configs.
---

# RAG eval / tuning loop

The instrument is `code/eval_harness.py`. One iteration changes **one** lever,
re-measures, and records the result. Never change two things at once — a mixed
result teaches nothing.

## Levers

| Lever | How to change it | Re-ingest? |
|---|---|---|
| retrieve-N (wide k) | `run_eval(label, wide_k=N)` | no |
| rerank top-k | `run_eval(label, top_k=K)` | no |
| hybrid retrieval | `run_eval(label, use_hybrid=True)` | no |
| planner breadth | edit the tagger prompt / label count in `code/subjects.py` | no |
| chunk size / overlap | `CHUNK_WORDS` / `OVERLAP_WORDS` in `code/ingest.py` | **YES** |
| query rewriting | add `rewrite_query()`, wire into `pipeline.run` before retrieve | no |

`run_eval(label, **kwargs)` forwards `top_k`, `wide_k`, `use_hybrid` to
`pipeline.run`. Anything else (chunking, planner, rewrite) is a code edit.

## The loop

1. **Know the baseline.** If there is no current baseline in this session, run
   one first:
   ```
   cd code && ../.venv/bin/python -c "from eval_harness import run_eval; run_eval('baseline (N=20, k=4)')"
   ```
   Run in the background — a full pass is ~2–4 min (17 questions x several
   Bedrock calls).

2. **Change ONE lever.** kwarg for retrieve-N / top-k / hybrid; code edit
   otherwise. State which lever and the hypothesis ("smaller chunks should lift
   completeness on benefits-and-perks.md").

3. **Re-ingest only if the lever is chunk config.**
   ```
   cd code && ../.venv/bin/python ingest.py --force
   ```
   `ingest.py --force` drops + recreates the index, then bulk-ingests. Real
   Bedrock spend (~165 embed + tag calls); replaces the shared `novaops-kb`
   index. Do NOT run it for any other lever.
   - Bare `ingest.py` (no flag) chunks cheaply first and re-ingests **only if
     the chunk count differs** from what's indexed — so after editing
     `CHUNK_WORDS` it often re-ingests on its own. But chunk count is a proxy:
     if you changed `OVERLAP_WORDS` and the count didn't move, use `--force`.
   - Add `create_index.py --force` first only if you edited `MAPPING` in
     `create_index.py` (ingest reuses that same object).
   - Prints `Bulk-indexed N/N chunks ... (M searchable)`; `M` may briefly lag
     `N` (AOSS eventually consistent) — recheck with `client.count()` before
     trusting an eval if it didn't converge.

4. **Re-measure.** Same command as step 1 with the new label + kwargs. For a
   hybrid change, run vector and hybrid back-to-back so the comparison is
   same-session:
   ```
   run_eval('vector — <change>', wide_k=20, top_k=4)
   run_eval('hybrid — <change>', wide_k=20, top_k=4, use_hybrid=True)
   ```

5. **Decide.** Compare the six aggregates (Recall@k, MRR, faithfulness,
   context-relevance, completeness, refusal accuracy) AND scan per-question
   rows for a single question that collapsed (a lever can lift the mean while
   destroying one case — see `MOON_PT` under hybrid).
   - **Keep** if a metric moved up and nothing regressed materially.
   - **Revert** the code edit / drop the kwarg otherwise. A lever that
     regresses but is worth keeping available (e.g. hybrid) stays **wired but
     opt-in** — never deleted.

6. **Record in `PROMPT.md`.** Update the relevant section's results block with
   a before/after table and one sentence naming which lever paid off (or why it
   didn't). Match the existing table style in §2 / §3.

7. **Update `advanced-rag-notes.md`** §5/§6/§9 if the finding changes the
   guidance there.

## Checks before you start

- `.env` valid: `cd code && ../.venv/bin/python -c "from env_check import validate_env; validate_env()"`
- Index populated: `create_index.py` no-ops and `ingest.py` reports an existing
  doc count. If empty, ingest once (step 3) before measuring.
- Only `client.py` and `judges.py` are provided — never edit them.
