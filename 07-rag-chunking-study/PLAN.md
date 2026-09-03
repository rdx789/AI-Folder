# Plan — Chunking Strategy Study

Source of truth for scope: `PROMPT.md` / `README.md`. This file is the plan and process
log, updated as work progressed. **The outcome — comparison table + recommendation — is
in `RESULTS.md`, not here.**

## Status: DONE

## 0. Setup
- Copied `../code/02-rag-as-tool/{ingest_kbs.py,rag_by_role.py,rag_tool.py,retrieval.py,tools.py}`
  into `homework/rag-as-tool/`. Originals in `../code` untouched.
- KB source files live at `../code/data/handbook` — read-only, NOT copied.

### Path wiring note
The copied files compute `CODE_DIR = Path(__file__).resolve().parents[1]`, which now
resolves to `homework/` instead of `code/`. Fine for index *output*
(`homework/data/faiss_index_*` stays local, matching the plan). Wrong for KB *input*, so
`chunking-study/common.py` hard-references the handbook source directly as
`Path(__file__).resolve().parent.parent.parent / "code" / "data" / "handbook"` (anchored
to script location per CLAUDE.md), independent of `CODE_DIR` in the copied files.

## 1. Chunking strategies (`chunking-study/`)
- `common.py`: config, `CallTracker` (embed calls, Nova calls/tokens, $ estimate),
  `load_handbook_files()`.
- `chunkers.py`: `chunk_fixed_size`, `chunk_separator` (heading-first then paragraph-split
  within a section; preamble/frontmatter before the first heading kept as its own
  section), `chunk_sentences`, `chunk_semantic` (batches of ≤40 paragraphs per Nova call —
  some handbook files have >100 paragraphs, so one-call-per-doc would've silently
  truncated; falls back to one chunk per batch on a bad/unparseable Nova reply).
- Every chunk tagged `{"text", "source"}` — filename carried through to indexing.

## 2. Indexing (`chunking-study/build_indexes.py`)
- Builds 8 variants — fixed×3 (300/600/800w, 50/100/200 overlap), separator×1,
  sentence×3 (3/5/8), semantic×1 — into `homework/data/faiss_index_<name>/`.
- Skips a variant if its index dir already has `index.faiss` (mirrors `ingest_kbs.py`'s
  idempotency).
- **Bedrock pricing** (checked 2026-08-09 via web search — no pricing constants existed
  in the repo already): Titan Text Embeddings V2 $0.00002/1K input tokens (no output
  tokens; Titan gives no usage field, so input tokens are estimated at ~4 chars/token);
  Nova 2 Lite $0.0003/1K input, $0.0025/1K output tokens (from usage field).
- Run with `code/.venv/bin/python` (faiss/boto3 aren't in the ambient interpreter, and no
  venv exists under `homework/` — reused the lab's venv rather than duplicating it).
- Real Bedrock run: all 8 indexes built successfully, ~$0.026 total. Per-strategy
  build stats are in `RESULTS.md`'s comparison table.

## 3. rag_tool.py updates (on the copy, `homework/rag-as-tool/`)
- `retrieval.py`: `load_kb`/`search` now carry `source` per chunk (4-tuple: id, score,
  text, source), backward-compatible with old plain-string chunk metadata.
- `tools.py`: `format_context` tags each chunk with its source file when known.
- `rag_tool.py`: SYSTEM prompt requires every factual claim to cite its source file
  inline (README calls this the "headline task", not a footnote).
- `rag_by_role.py`: `print_scores` updated for the new 4-tuple (kept runnable; out of
  PROMPT.md's explicit scope but would've crashed otherwise).

### Citation-duplication fix
Smoke-testing `python rag_tool.py` after shipping the default index surfaced duplicate
citations like `(managing-work-devices.md)⟨zero-width⟩ (managing-work-devices.md)`.
Root cause, confirmed by inspecting the raw Converse response: Nova fires its own native
citation-grounding markup *in addition to* the explicit `(file.md)` text the prompt asks
for — not a prompt-compliance failure. Tightening the prompt wording didn't change the
behavior (tried and reverted). Fixed by sanitizing the answer instead: `rag_tool.py` now
strips zero-width chars (U+200B/200C/FEFF) and collapses back-to-back duplicate
`(file.md)` citations before returning the answer. Re-verified — citations are single and
clean.

## 4. Eval question set (`chunking-study/eval_questions.py`)
Built per README's required mix: 8 single-fact lookups, 2 cross-section-of-one-document
questions, 2 unanswerable/refusal checks. Every `expected_source` and keyword group
verified against actual handbook text before locking. Keyword groups are used for
automatic correctness grading (a pragmatic stand-in for grading 96 answer cells by hand).

## 5. Eval harness (`chunking-study/run_eval.py`)
Runs each of the 12 questions through the **actual** `rag_tool.py` agent loop (monkeypatches
`tools.retrieve` only to observe results, not to change behavior) against each of the 8
indexes in turn, by pointing `tools.KB_REGISTRY`'s `search_handbook` entry at each
strategy's index dir. Logs retrieved chunk ids + cosine + source + final cited answer per
question; writes `eval_results.json`.

Caught and fixed one grader bug during review: the refusal-phrase list for the 2
unanswerable questions initially missed phrasings like "can't share" / "I can only
access", which undercounted every strategy's correctness by 1, uniformly. Fixed the
phrase list and rescored from the saved answers (no Bedrock re-run needed).

## 6. Scoring & comparison
Right-source hit rate (top-k included a chunk from the expected file) and keyword-based
answer correctness, per strategy — table and recommendation written up in `RESULTS.md`.

## 7. Shipped default
`homework/data/faiss_index_handbook/` holds a copy of `faiss_index_fixed_600w_100ov`
(same vectors, no re-embedding needed) so `python rag_tool.py` run directly, no extra
setup, uses the winning config. Verified via `tools.load_tools(["search_handbook"])` and
two live smoke-test questions.

## Decisions log
- Eval questions: proposed by Claude, approved by user before running (not guessed
  silently); structured per README's single-fact / cross-section / unanswerable mix.
- Pricing: checked repo first (none existed), fetched current published Bedrock pricing
  via web search.
- Helpers: copied `02-rag-as-tool` into `homework/rag-as-tool/`, edited the copies, kept
  `../code` untouched.
- No `fixed_250w_50ov` control index added — user call: 300w already serves as the
  fixed-size sweep's low end, no functionally different implementation needed.
- Manager-playbook repeat-study and further overlap tuning: left as optional next steps
  (README marks both optional) — see `RESULTS.md`'s "Possible next iteration".

## Files produced
`chunking-study/{common.py,chunkers.py,build_indexes.py,eval_questions.py,run_eval.py,
eval_results.json}`; modified copies in `homework/rag-as-tool/{retrieval.py,tools.py,
rag_tool.py,rag_by_role.py}`; 8 indexes under `homework/data/faiss_index_*/` plus
`homework/data/faiss_index_handbook/` (the shipped default); `RESULTS.md` (deliverable).
