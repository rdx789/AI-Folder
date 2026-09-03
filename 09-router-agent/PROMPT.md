GOAL: A top-level router for the NovaOps assistant — read any request, classify it,
and route it to exactly one handler.

Clean single-path routing: one request, one handler. No fan-out, no merging, no
`Send`, no reducers — the deliberate opposite of the parallel supervisor. The full
supervisor build (plan → parallel specialists → reflection → approval gate) is the
sibling `09-parallel-supervisor/` project; this one does not use it.

## Constraints

- `server/` is PROVIDED as the finished NovaOps MCP server. Start it, connect over
  MCP, and leave it alone.
- Discover the server's tools at runtime; let `langchain-mcp-adapters` turn them into
  framework tools. Keep the MCP SDK pinned `mcp<2`.
- All config from the environment via `load_dotenv(find_dotenv())`. Reach Bedrock
  through one small factory.

## Files

- **`config.py`** — `load_dotenv(find_dotenv())`, check the five required vars
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`,
  `BEDROCK_EMBEDDING_MODEL_ID`) at import and raise `RuntimeError` on any missing.
  `get_llm(temperature=0.0)` is the single `ChatBedrockConverse` construction point.
  Owns `MCP_SERVER_URL` (default `http://127.0.0.1:9876/mcp`, `NOVAOPS_MCP_URL`
  override).
- **`mcp_client.py`** — `MultiServerMCPClient` against `MCP_SERVER_URL`
  (`streamable_http`); `discover_tools()` → `{tool_name: BaseTool}`, clear
  `RuntimeError` if unreachable.
- **`router.py`** — the router graph, its `RouterState`, and a CLI.

## Graph shape

```
START ─▶ classify ──(route on category, exactly one)──┬─▶ policy_qa       ─▶ END
                                                      ├─▶ it_ticket       ─▶ END
                                                      ├─▶ lookup          ─▶ END
                                                      ├─▶ access_request  ─▶ END
                                                      └─▶ general         ─▶ END
```

`RouterState`: `request`, `category`, `handler`, `answer`.

- **`classify`** — `llm.with_structured_output(Category)` labels the request as one
  of `policy_qa` / `it_ticket` / `access_request` / `lookup` / `general`. Writes
  `state["category"]` only. On any failure, label it `general`.
- **route** (conditional edge) — a plain function returning `state["category"]`, so
  exactly one handler node runs.
- **`policy_qa`** — focused `create_agent` over `list_policies` + `get_policy`.
- **`it_ticket`** — focused `create_agent` over `search_knowledge_base`.
- **`lookup`** — focused `create_agent` over `get_employee`.
- **`access_request`** — focused `create_agent` over `get_employee`,
  `check_software_subscription`, `list_policies`, `get_policy`,
  `create_access_request`. It identifies the employee, checks seat availability and
  policy, and files the request only if the employee is clearly identified and
  eligible; otherwise it asks for the employee id.
- **`general`** — a plain deterministic node: a canned "here's what I can help with"
  reply. An agent would be overkill.

Every handler writes `handler` + `answer` and goes straight to `END`.

Tracing: a `trace(step, detail)` helper prints `[trace] <step> -> <detail>`.
`classify` logs the chosen category and its reason; the handler that runs logs one
line. Skipped handlers log nothing, so the output is the real single path taken.

## Run

Terminal 1 — start the provided server:

    python server/server.py     # serves .../mcp on 127.0.0.1:9876

Terminal 2 — route the three sample requests to three different handlers:

    python router.py
    #  "what's the expense policy?" -> policy_qa
    #  "I need Webex access"        -> access_request
    #  "who is E010?"               -> lookup

Route and answer a single request:

    python router.py "how do I reset my MFA?"
