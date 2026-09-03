GOAL: Build a NovaOps assistant that runs a team of agents, with a human on the write.

Method: a LangGraph supervisor over specialist sub-agents, with a human approval
gate in front of the one write action.

## Constraints

- `server/` is PROVIDED as the finished NovaOps MCP server. Start it, connect over
  MCP, and leave it alone: don't edit it, don't reimplement its tools in the build.
- Read what the server actually offers BEFORE designing anything. Its tool set is
  what decides how the work can be divided.
- Build on LangChain / LangGraph; let `langchain-mcp-adapters` turn the discovered
  MCP tools into framework tools (no hand-written schemas). Keep the MCP SDK pinned
  `mcp<2` — `langchain-mcp-adapters` doesn't support the 2.0 SDK yet.
- All config from the environment via `load_dotenv(find_dotenv())`. Reach Bedrock
  through the framework's chat-model class, kept in one small factory.

## Files to build

- **`config.py`** — call `load_dotenv(find_dotenv())`, then at import time check the
  five required vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
  `BEDROCK_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID`) and raise `RuntimeError` listing
  any that are missing — stop rather than guess. Expose `get_llm(temperature=0.0)`
  as the single `ChatBedrockConverse` (`langchain_aws`) construction point so the
  provider stays swappable. Own `MCP_SERVER_URL` here too — default
  `http://127.0.0.1:9876/mcp`, overridable via `NOVAOPS_MCP_URL`.
- **`mcp_client.py`** — a `MultiServerMCPClient` against `MCP_SERVER_URL` (transport
  `streamable_http`); `discover_tools()` returns `{tool_name: BaseTool}` discovered
  live. Raise a clear `RuntimeError` if the server is unreachable.
- **`specialists.py`** — three `create_agent()` sub-agents, each bound only to its
  assigned subset of the discovered tools (never hand-written schemas). Split the
  work along what the server offers, e.g.:
  1. `knowledge` → knowledge-base search
  2. `records` → employee lookup, software-subscription check
  3. `eligibility` → employee lookup, policy listing and retrieval
  End every specialist prompt with a shared work-bounding clause: at most three tool
  calls, then a 2-3 sentence answer.
- **`supervisor.py`** — `build_graph(tools_by_name, checkpointer)` returns the
  compiled LangGraph graph; also defines `SupervisorState` and the `trace` helper.
  The checkpointer is passed in (not created here) so the caller can hold the async
  saver open for the whole run.
- **`main.py`** — the entry point.
  - `python main.py "<request>"` runs the whole flow in one process: it pauses
    inline at the write proposal for y/n (plus a reason on n), then finishes. With
    no request, use a `DEFAULT_REQUEST`. A request with no write proposal just runs
    to the answer.
  - Each run gets its own `thread_id`; it's stashed in a sidecar file so
    `python main.py resume` can recover the last run if its process was killed
    while sitting at the approval prompt — it reopens that exact on-disk checkpoint
    and continues.
  Demonstrate the reflection loop with an under-specified request that triggers a
  second round and a clear one that finishes in the first.

## Graph shape

```
plan ──(dispatch)──▶ worker  (Send, one per chosen specialist — one parallel superstep)
           │  (fan-in barrier: waits for every branch; operator.add reducer
           │   merges each branch's one-element specialist_outputs list)
           ▼
        review ──┬──▶ (need_more, round < 2) ──▶ worker  (re-dispatch the same
                 │                                        specialists via Send)
                 └──▶ (enough) ──▼
        propose_write ──┬──▶ finalize                     (no write needed)
                        │
                        └──▶ approval_gate ──┬──▶ execute_write ──▶ finalize
                                             └──▶ finalize          (denied)
```

- **`plan`** — use `llm.with_structured_output(Plan)` to choose the specialist
  subset for THIS request over a closed `Literal["knowledge","records","eligibility"]`
  set (also capture a one-line `reasoning`). Fall back to `["knowledge"]` if the
  model returns an empty list or the call fails.
- **`dispatch`** (conditional edge) — return
  `[Send("worker", {...}) for name in plan.specialists]`, so the chosen specialists
  run concurrently in a single superstep, not a for-loop. The specialists have no
  data dependency and only merge at the end, so concurrency is a free latency win.
- **`worker`** — run one specialist sub-agent to completion and return
  `{"specialist_outputs": [{"agent": name, "summary": ..., "round": ...}]}`. Put an
  `operator.add` reducer on that state key to make the fan-out race-free; the
  `round` tag lets `review` and `finalize` read only the latest round's outputs.
- **`review`** — add a `review` node after `worker` that uses the model to decide
  `enough` vs `need_more`, and add `round` and `follow_up_request` to the state. On
  `enough`, go to `propose_write`. On `need_more` — and only while `round` < 2 —
  increment `round`, write a refined `follow_up_request`, and re-dispatch the same
  specialists selected in round one with that follow-up.
- **`propose_write`** — use `llm.with_structured_output(WriteProposal)` to decide,
  FROM the specialists' findings (not the raw request), whether resolving this needs a
  `create_access_request` — a WRITE. Downgrade to "no write" unless the proposal has
  a well-formed `employee_id`, a `business_justification`, and a single concrete
  `software` that the user actually named (reject placeholder-looking values like
  `SOMETHING_LIKE_THIS` / "unknown" / "tbd" / a slash-joined list, and reject a
  `software` that appears nowhere in the request — the model sometimes invents a
  write for a plain lookup). Read-only requests route straight to `finalize` and
  never interrupt.
- **`approval_gate`** — call `interrupt()` to surface the proposed write to the
  caller. The caller resumes with `{"approved": bool, "reason": str|None}`. A denial
  with no reason raises `ValueError` (hard rule, not a UI convention).
- **`execute_write`** — the ONLY code path that calls the real `create_access_request`
  MCP tool; wrap it in try/except.
- **`finalize`** — merge the specialists' findings into one answer. Tell the model
  the write outcome so the prose matches it: if a request was filed, name its
  reference and say no further action is needed (don't suggest a separate IT
  ticket); if it was denied, say access was not granted and why. Append a short
  machine-written status line after the answer.

Durable pause: use `AsyncSqliteSaver`
(`from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`; `pip install
langgraph-checkpoint-sqlite` if it's missing) writing to a local file, with a
per-run `thread_id` in the LangGraph run config
(`{"configurable": {"thread_id": ...}}`). The normal run is one process — invoke,
service the `interrupt()` inline, then invoke again with `Command(resume=...)`. The
on-disk checkpoint is what makes `resume` possible: if the process is killed while
waiting at the approval prompt, a later `resume` reopens that checkpoint and
finishes. The graph is invoked asynchronously (`ainvoke`), which is why the saver
must be the async one.

Tracing: give the run a single `trace(step, detail)` helper that prints
`[trace] <step> -> <detail>`. Every node emits at least one line as it runs, so the
output is the real, ordered path to the answer:

- `plan -> specialists=[...] (<planner reasoning>)`
- `dispatch -> sending to N specialist(s) in parallel: [...]`, or
  `dispatch -> re-dispatching to N specialist(s): [...]` on a reflection round
- `worker -> '<name>' running...` then `worker -> '<name>' done (<n> chars)`
- `review -> enough (round N)` or `review -> need_more (round N): <follow-up>`
- `propose_write -> WRITE needed: create_access_request(employee_id=..., software=...)`
  or `propose_write -> no write needed`
- `approval_gate -> APPROVED` / `DENIED: <reason>`
- `execute_write -> calling create_access_request for <id>...` then `-> done: <result>`
- `finalize -> merging the specialists' findings into final answer`

Nodes that are skipped for a given request produce no line.

## Write-gating

Make the write decision in a separate `propose_write` structured-output step AFTER
the specialists report back — the model never emits the `create_access_request` tool
call itself. Validate the args before the human sees them, and make `execute_write`
the single code path that calls the real tool. Gate before write, require a reason
on reject, return a clean answer either way.

## Run

Terminal 1 (must be up first) — start the provided server:

    python server/server.py     # serves .../mcp on 127.0.0.1:9876

Terminal 2 — one process: it pauses at the write proposal for `y` / `n` (`n`
requires a non-empty reason), then finishes:

    python main.py "I'm E010 and Webex says I'm not licensed — can you help?"

If that process is killed while waiting at the prompt, recover it:

    python main.py resume