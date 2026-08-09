GOAL: Build a RAG tool as an agentic RAG that can be utilized by any app.

The project consists of a `data/` folder that holds one subfolder per knowledge
base, each full of `.md` files. Assume more subfolders (more knowledge bases)
will be added later without code changes — discover them at runtime instead of
hard-coding names.

Use a `.env` file to define OS environment variables and read them at runtime;
never hard-code region, model IDs, or credentials.

## Files to build

- `bedrock_client.py` — single module owning every Bedrock call, so a provider
  swap only touches this file:
  - `converse(messages, system=None, tool_config=None, tool_choice=None, ...)` —
    wraps `bedrock-runtime.converse()` for both plain chat and tool-use turns.
  - `embed(text, dimensions=None)` — wraps `invoke_model()` against the Titan
    embedding model (`inputText`, `dimensions`, `normalize: True`), returns
    `list[float]`.
  - Loads config via `load_dotenv(find_dotenv())`; reads `AWS_REGION`,
    `BEDROCK_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID` from env (required), and
    `BEDROCK_EMBEDDING_DIMENSIONS` (optional, default `1024`).
  - Exposes `DATA_DIR = Path(__file__).resolve().parent / "data"` so every
    other module anchors paths to the script location, not the shell's cwd.

- `registry.py` — auto-discovers knowledge bases by scanning `DATA_DIR` for
  subfolders (no hard-coded KB list). For each subfolder `data/<name>/`:
  - `name` = folder name.
  - `description` = contents of `data/<name>/_description.txt` if present,
    else a generic fallback string derived from the folder name. The
    description is what the LLM sees when deciding which KB to search, so
    write a real `_description.txt` per KB (audience, topics covered, topics
    explicitly NOT covered) for good routing.
  - `index_dir` = `data/<name>/index/`, holding `index.faiss` and
    `chunks.json` (`{"embedding_model": <id>, "chunks": [{"text":, "source":}, ...]}`
    — the embedding-model id rides alongside the vectors so a query can be
    checked against the model that built the index).
  - `get_registry() -> dict[str, KnowledgeBase]` returns the name → KB map.

- `ingest.py` — dedicated ingestion script, run manually (`python ingest.py`)
  whenever `data/` content changes:
  - For every KB in the registry, glob `*.md`, chunk each file (paragraph-
    aware, ~1000 chars per chunk with ~150 char overlap — see `chunk_text()`),
    embed every chunk via `bedrock_client.embed`, and build a FAISS
    `IndexFlatIP` (embeddings are normalized, so inner product = cosine).
  - Write `index.faiss` and `chunks.json` (with the embedding-model id, see
    above) into `data/<kb>/index/`.
  - Index build is a manual, explicit step — the app does NOT auto-build a
    missing index; `search.py` raises a clear error telling the user to run
    `python ingest.py` if the index is missing.

- `search.py` — retrieval helper used by the agent:
  - Caches loaded FAISS indexes + chunk records per KB in-process.
  - On load, asserts `chunks.json`'s recorded `embedding_model` matches the
    live `BEDROCK_EMBEDDING_MODEL_ID` — raises a clear "re-run ingest.py"
    error on mismatch, so a query never gets embedded with a different model
    than the documents were.
  - `search(kb, query, top_k=4) -> list[dict]` embeds the query, does a
    FAISS similarity search, returns chunk records with a `score`.

- `rag_tool.py` — the agentic RAG tool + app entry point:
  - Defines **one generic Bedrock tool spec**, `search_knowledge_base`, with
    `kb_name` (enum built from the live registry) and `query` string
    parameters. Do not generate one tool per KB — a single generic tool that
    takes a KB name scales to any number of knowledge bases without new tool
    specs.
  - **Retrieval is model-decided, not forced.** Every `converse()` call in
    the loop uses `tool_choice={"auto": {}}`, so the model itself decides
    whether a question needs a search at all — greetings and "what can you
    do" questions get answered directly, with zero tool calls; policy/fact
    questions trigger one or more `search_knowledge_base` calls. Do NOT force
    the tool choice — a forced-tool design makes the model search even on
    "hello," which defeats the point of agentic RAG.
  - `answer_question(question) -> str` — a proper multi-turn **agent loop**
    (not a single plan/execute/answer pass), capped at `MAX_TURNS = 5`:
    - Call `converse()` with the running `messages` list and the tool config.
    - If `stopReason != "tool_use"`, return the model's text — done.
    - Otherwise, the assistant turn may contain **more than one** `toolUse`
      block in a single turn (e.g. a question spanning two knowledge bases).
      For **every** `toolUse` block, run `execute_tool()` and append one
      matching `toolResult` (same `toolUseId`) to the next user message —
      Converse rejects the history if any `toolUse` is left without a
      `toolResult`. Print a one-line trace of each search (`kb`, `query`) so
      the tool-gating behavior is visible when running interactively.
    - Loop back to `converse()` with the updated `messages`.
    This is the function any other app should import and call.
  - `execute_tool(tool_use, registry)` — runs `search.search()` against the
    KB and query the model chose for one `toolUse` block.
  - `main()` — a REPL: prompts for a question in a loop, prints the answer,
    exits on `quit`/`exit`/EOF/Ctrl-C. (Not a single ask-once-and-exit app —
    keep looping so the agent can be tested interactively.)

- `requirements.txt` — `boto3`, `python-dotenv`, `faiss-cpu`, `numpy`.

## Design decisions (already made — do not re-ask)

- Registry shape: **one generic search tool** + a `kb_name` parameter chosen
  from the registry, not one Bedrock tool per KB.
- App loop: **REPL** (`main()` keeps asking until the user quits), not a
  single question-and-exit.
- Index storage: `data/<kb>/index/index.faiss` + `data/<kb>/index/chunks.json`
  (with the embedding-model id recorded alongside the chunks), built by a
  **manual** `python ingest.py` run. The app errors with an actionable
  message if an index is missing, or if it was built with a different
  embedding model than the one currently configured, rather than silently
  proceeding or auto-building.
- Tool choice: **`{"auto": {}}`**, never forced. The model decides per-turn
  whether to call `search_knowledge_base`, so chit-chat skips retrieval
  entirely. `answer_question()` is a real loop (capped at `MAX_TURNS = 5`)
  that resolves every `toolUse` block emitted in a turn — including more
  than one in the same turn — before calling `converse()` again.

## Adding a new knowledge base later

1. Create `data/<new_name>/` with `.md` files.
2. Add `data/<new_name>/_description.txt` describing its audience/scope.
3. Run `python ingest.py` (rebuilds indexes for every KB found, including
   existing ones).
4. No code changes needed — `registry.py` picks it up automatically, and the
   agent can now route questions to it.

## Conventions (from this repo's CLAUDE.md)

- Call the model with boto3 Bedrock Converse.
- Keep the Bedrock call in one function so a provider swap is easy.
- Config from env: `.env` via `load_dotenv(find_dotenv())`; model IDs + region
  from env vars, never hard-coded.
- Anchor file paths to the script location, not the shell's cwd.
- Comment only the non-obvious AI/SDK bits — skip the obvious.