---
name: validate-lab
description: Validate a student's finished rebuild of Lesson 6's tool-based RAG assistant (rag_tool.py) against the reference lab. Scores the build out of 100 (can exceed 100 if it beats the reference), reports what's missing, weaker, or better per check, and offers a cheat prompt to close the gap.
---

# Validate Lab — Lesson 6, tool-based RAG (`rag_tool.py`)

Score a finished `rag_tool.py` (plus its ingest/retrieval helpers) against the
reference build — out of 100 — and report where it's missing, weaker, or better.

## What you're validating

The student's job was to build an assistant over the NovaOps employee handbook
where **retrieval is a tool the model decides when to call** — not the
retrieve-on-every-turn pipeline of `01-simple-rag`. The load-bearing ideas:

1. **Ingest** — chunk the handbook, embed each chunk, and store it so it can be
   searched. The chunk **texts** and the **embedding-model id** ride *alongside*
   the vectors (a vector index holds only vectors; a search result is just a row
   number), and the query must be embedded with the **same** model the documents
   were.
2. **Retrieval as a tool** — a `search_handbook` tool with a `toolChoice: auto`
   config, so the model calls it for policy questions and skips it for "hello."
   The tool's **description** is what makes that routing work.
3. **The agent loop** (Lesson 4's loop, now with a retrieval tool) — every
   `toolUse` block the model emits gets a matching `toolResult` in the next user
   message (or Converse rejects the history), it loops until the model stops
   asking to search, and it caps the turns.
4. **Grounding** — a system prompt that makes the model answer **only** from what
   it retrieved and say so when the answer isn't there (this corpus is adapted
   from a public handbook, so an ungrounded model will happily answer from
   memory and hide the bug).

The student may have split the files differently (one file, or ingest / retrieval
/ tool split like the reference) and picked different chunk sizes, top-k, tool
name, or prompt wording — judge the **shape and behavior**, not the exact labels.

## Locate the two builds

You're running from the lesson's **reference repo**, inside its `SDD/` folder — so
the answer key is local (`../code/`) and the student's build lives in a separate repo.

- **Student build:** the student built in a **separate** clean repo (the `…-sdd`
  repo they cloned). **Ask them for the path to it**, then read
  `<their-path>/rag_tool.py` plus whatever ingest/retrieval helper files it
  imports. If they've `cd`'d into that repo the path may just be `.`; if there's
  more than one candidate `.py`, ask which is the runnable assistant.
- **Reference build:** `../code/02-rag-as-tool/rag_tool.py`, with its helpers
  `../code/02-rag-as-tool/retrieval.py`, `tools.py`, and `ingest_kbs.py`, sitting
  beside this `SDD/` folder in the same reference repo. **Ignore the role/plan
  machinery** in `tools.py` and `ingest_kbs.py` (`ROLE_TOOLS`, `plan_tool`,
  multi-corpus discovery) and `rag_by_role.py` — that's a *different* exercise,
  out of scope here. Compare only against the single-tool `rag_tool.py` slice.
  If `../code/` isn't there, validate structure-and-behavior only and skip the
  semantic comparison.

---

## Check 1 — Structural comparison (45 pts)

Read the student's files and confirm the load-bearing pieces are present.

- **Retrieval wrapped as a tool, model-decided (12):** a search tool with a
  `toolSpec` (a `query` string input) handed to `converse` via
  `toolConfig` with `toolChoice: {"auto": {}}` — so the *model* chooses whether
  to search. A version that retrieves unconditionally on every turn (the
  `01-simple-rag` pipeline) misses the entire point of this lab — score near zero
  here.
- **A correct agent loop (13):** loops on `stopReason == "tool_use"`; for **every**
  `toolUse` block in the model's message it appends a `toolResult` carrying the
  matching `toolUseId` (handling *more than one* search in a single turn); returns
  the model's text once it stops asking to search; and caps iterations
  (a `MAX_TURNS`-style guard). A loop that answers only the first `toolUse`, or
  reuses/omits a `toolUseId`, is the headline bug — dock heavily.
- **Grounding system prompt (10):** a system prompt that (a) tells the model to
  search for policy/benefit/how-the-company-works questions but answer greetings
  and capability questions directly, and (b) constrains the answer to the
  retrieved context, with an explicit "say so if it's not there." No grounding
  rules → the model answers from training data.
- **Ingest that stores texts + model id (6):** documents are chunked, each chunk
  embedded, and a vector index built — **with the chunk texts and the
  embedding-model id persisted alongside the vectors**, not thrown away. (A build
  that stores only vectors can't return text; one that doesn't record the model
  can't guarantee the query is embedded the same way.)
- **Same-model query embedding + top-k (4):** the query is embedded with the same
  model/dimensions the documents were, and retrieval returns a bounded top-k.
- **Conventions (bonus, up to +4):** model IDs / region from env via
  `find_dotenv()`, the Bedrock call isolated in one function, paths anchored to
  the script rather than cwd.

## Check 2 — Deterministic behavior (30 pts)

Retrieval and the loop invariant are testable **without** judging generated prose.

**2a. Retrieval returns the right chunks (18).** Build (or load) the student's
index, then run their retrieval function on a few queries with knowable answers
and confirm the top chunk is on-topic:

- `"can I work a second job?"` → the **moonlighting** policy (the marquee
  semantic-search case: it matches with **no shared keywords**);
- `"how many weeks of paid vacation do I get?"` → the benefits / time-off doc;
- `"what happens to my laptop when I leave?"` → the device / severance material.

If their retrieval is importable, call it directly; otherwise drive their ingest
+ a one-shot search via a tiny script. A retrieval that returns off-topic chunks
(or degenerates to keyword matching and misses "second job" → moonlighting) fails
here.

**2b. The loop answers every toolUse (12).** Confirm the agent loop returns a
`toolResult` for **each** `toolUseId` the model emits. Prefer reading the loop;
if you want to prove it, drive one turn (a policy question) and inspect the
`messages` list — the user message after a two-`toolUse` turn must contain two
`toolResult` blocks with the two matching ids. A missing/duplicated id is the
bug that makes Converse 400 on the next turn.

## Check 3 — Semantic comparison of model output (25 pts)

Run **both** builds end-to-end (they read questions from stdin — pipe them in).
This makes real Bedrock calls — needs a working `.env` and a built index (e.g.
`source setup.sh`, then run the ingest step; the reference index is under
`../code/data/faiss_index_handbook/`, rebuild with
`python ../code/02-rag-as-tool/ingest_kbs.py` if absent). If creds aren't
available, say so and skip this check. Drive each with the same inputs, e.g.
`printf 'hello\nHow much paid vacation do I get?\nWhat is our AWS bill?\nquit\n' | python <build>/rag_tool.py`, and confirm:

- **Greeting → no search.** "hello" is answered directly, with **no** retrieval
  fired (the reference prints a `-> search_handbook(...)` line only when it
  searches — the student's build should likewise not search here). A build that
  searches on a greeting has lost the tool-gating behavior.
- **Policy question → search + grounded answer.** "How much paid vacation do I
  get?" fires a handbook search and the answer reflects the retrieved policy,
  matching the reference semantically (same figure / same policy), not invented.
- **Unanswerable question → refuses gracefully.** "What is our AWS bill?" (not in
  the handbook) → the model says it can't find that in the handbook rather than
  fabricating a number. Compare to the reference's behavior.

A build whose answers are individually fluent but that searches on greetings, or
answers policy questions from training data instead of retrieval, is weaker even
if the prose reads fine.

---

## Scoring and report

Total the three buckets into a score **out of 100** — and **let it exceed 100**
when the student's build is genuinely better than the reference, e.g.:

- persists source filenames per chunk and cites them in the answer;
- a cleaner ingest/retrieval/tool split, or a retrieval layer parameterized by
  index so a second corpus is a one-liner;
- a calibrated "nothing scored high enough → refuse" guard on top of the grounding
  prompt;
- swaps FAISS for a persistent vector DB (Chroma) while keeping the same search
  signature.

Report back, concretely and anchored in *this* lab (never a generic diff):

1. **Score** (e.g. `86 / 100`, or `108` if it beats the reference) with a
   one-line justification per check.
2. **What's missing or weaker**, per check — name the exact gap ("your loop only
   handles the first `toolUse` block, so a two-part question drops the second
   search and 400s on the retry"; "retrieval is unconditional — the model never
   gets to *decide* to search, which is the whole lesson"; "no grounding rule, so
   'What is our AWS bill?' came back with an invented figure").
3. **What's better than the reference**, if anything — say so explicitly.

Then **offer a cheat prompt** the student can paste to bring their build up to (or
past) the reference — but tell them to use it **only if they're out of time**.
Name the specific gaps you found, e.g.:

> "Refactor into `ingest.py`, `retrieval.py`, and `rag_tool.py`. In `ingest.py`,
> chunk each handbook doc with a sliding window (250 words, 50 overlap), embed
> each chunk with Titan (`invoke_model`, `normalize=True`), build a FAISS
> `IndexFlatIP`, and save the chunk texts + embedding-model id in a
> `metadata.json` beside the index. In `retrieval.py`, load that and expose
> `search(query, top_k=4)` that embeds the query with the recorded model and
> returns `(chunk_id, cosine, text)`. In `rag_tool.py`, define a `search_handbook`
> toolSpec (single `query` field, a description naming what the handbook covers),
> call `converse` with `toolChoice: {"auto": {}}` in a loop: on `stopReason ==
> 'tool_use'`, for every `toolUse` block run the search and append a `toolResult`
> with its `toolUseId`; stop otherwise, capping at 5 turns. System prompt: search
> for policy/benefits/company questions, answer greetings directly, and answer
> ONLY from retrieved context — say so if it's not there."
