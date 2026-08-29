# How the MCP server works — flow and usage

This explains what the pieces in `multi-tool-server/` are, how a question flows
through them, and how to run the whole thing.

---

## 1. The three pieces

| File | Role | Changes when you add a tool? |
|------|------|------------------------------|
| `server/server.py` | **MCP server** — declares the tools, holds the transport | **yes** — this is the only file that grows |
| `agent.py` | **MCP client + agent loop** — connects, discovers tools, runs the Bedrock tool-use loop | no |
| `client.py` | **provider bridge** — Bedrock Converse client + MCP→Bedrock schema translation | no |

The server and the agent are separate processes. They talk over HTTP using the
**MCP streamable-HTTP transport** (`http://127.0.0.1:9876/mcp`). Everything the
agent knows about the tools — their names, argument schemas, descriptions — it
learns at runtime by asking the server. Nothing is hard-coded on the agent side.

```
┌─────────────┐   MCP over HTTP    ┌──────────────────────┐
│  agent.py   │ ◄───────────────►  │   server/server.py   │
│  (client)   │  list_tools        │   @mcp.tool() x9     │
│             │  call_tool         │                      │
│  Bedrock    │                    │   ├─ policy files    │  data/policies/*.md
│  tool-use   │                    │   ├─ rag.py ─────────┼─ OpenSearch Serverless
│  loop       │                    │   └─ db.py ──────────┼─ in-memory SQLite
└─────────────┘                    └──────────────────────┘
       │
       ▼
  Bedrock Converse  (Nova model decides which tool to call)
```

---

## 2. The server side (`server/server.py`)

### Declaring a tool

```python
mcp = MCPServer("novaops-assistant")

@mcp.tool()
def check_asset_inventory(asset_type: str = "", location: str = "") -> list[dict]:
    """Check hardware inventory — what spare equipment is in stock and unassigned.
    ...
    Args:
        asset_type: The kind of asset, e.g. 'Laptop' or 'Monitor'. Optional.
        location: Where the stock should be, e.g. 'United Kingdom'. Optional.
    """
    return db.check_asset_inventory(asset_type, location)
```

The `@mcp.tool()` decorator turns a plain Python function into an MCP tool. The SDK
builds the tool's **JSON Schema** from the type hints (`asset_type: str = ""` →
optional string) and its **description** from the docstring — including the `Args:`
section, which the model reads to decide what to pass. That is the entire tool
definition; there is no separate schema file.

Each tool body is a thin wrapper that calls into a backend module:

- `list_policies` / `get_policy` — read `data/policies/*.md` off disk
- `search_knowledge_base` → `rag.search_kb()` — k-NN vector search over OpenSearch
- `get_employee`, `check_software_subscription`, `check_asset_inventory`,
  `list_employee_tickets` → `db.*` — SQL **reads**
- `create_ticket`, `create_access_request` → `db.*` — SQL **writes**

The agent cannot tell these apart — a file read, a vector search, and a SQL insert
are all just "a tool with a name and a schema." That uniformity is the point of MCP.

### The backends

- **`server/db.py`** — builds an in-memory SQLite DB once from
  `data/database/schema.sql` + `seed.sql` (`_conn()`, `lru_cache`d). One shared
  connection, so a `create_ticket` write is visible to a later
  `list_employee_tickets` read in the same server run. Writes are lost on restart.
- **`server/rag.py`** — `search_kb(query)` embeds the query with Titan
  (`kb_client.embed_text`) and runs a k-NN query against the OpenSearch Serverless
  index `novaops-kb`, returning `[{"article", "score", "text"}, ...]`. The
  embedding call happens **here, on the server** — capability and cost live
  server-side, not in the agent.
- **`kb_client.py`** — the OpenSearch + Bedrock clients (`opensearch_client()`,
  `embed_text()`), resolving the collection endpoint from `OPENSEARCH_COLLECTION`.

### Startup

```
python server.py
  → rag.warm_index()          # HEAD + _count on the novaops-kb index — fail loudly now, not on Q1
  → prints "KB index reachable — 27 chunks searchable."
  → mcp.run(transport="streamable-http", host="127.0.0.1", port=9876)
  → serves MCP at http://127.0.0.1:9876/mcp until Ctrl-C
```

---

## 3. The client side (`agent.py`) — the request flow

### Connect + discover (once per session)

```python
async with Client(MCP_SERVER_URL) as client:      # opens HTTP conn + runs the MCP `initialize` handshake
    tools = (await client.list_tools()).tools     # server returns all 9 tool defs (name, schema, description)
    tool_config = {"tools": mcp_tools_to_bedrock(tools)}   # translate MCP shape → Bedrock `toolConfig` shape
```

`mcp_tools_to_bedrock()` in `client.py` is the only provider-specific glue: it
rewraps each MCP tool as `{"toolSpec": {"name", "description", "inputSchema": {"json": <schema>}}}`.
Driving the same server from OpenAI or Anthropic would swap only this function.

### The tool-use loop (once per question) — `answer()`

```
messages = [user question]
repeat up to MAX_TURNS:
    response = bedrock.converse(model, system=SYSTEM_PROMPT, messages, toolConfig)

    if stopReason != "tool_use":
        return the model's text          # end_turn — the model is done

    for each toolUse block the model emitted:
        result = await client.call_tool(name, input)     # ← runs over MCP, on the server
        append a toolResult block (with status:"error" if result.is_error)
    append all toolResults as the next user turn
    loop
```

So one question can round-trip several times: the model asks for
`get_employee`, reads the result, then asks for `create_ticket`, reads that, then
writes a final answer. `run_tool_calls()` is the half that actually touches the
server; `answer()` is just the turn-taking around it.

`result_text()` joins **every** content block in a tool result — a tool that
returns a list comes back as one text block per item, so reading only the first
block would silently drop the rest.

### End-to-end, concretely

```
$ python agent.py "File a ticket for E010: can't connect to the VPN. Category Network."

  bedrock.converse(...)                    → model wants create_ticket
  → MCP tool: create_ticket({'employee_id':'E010','subject':...,'category':'Network'})
      server.py  create_ticket()           → db.create_ticket()
        db.py    checks E010 exists, INSERT TCK-0001 (status open, team IT), returns the row
  toolResult: {ticket_id: "TCK-0001", status: "open", assigned_team: "IT", ...}
  bedrock.converse(...)                    → stopReason end_turn
  prints: "Your support ticket has been filed successfully: Ticket ID TCK-0001 ..."
```

---

## 4. Running it

### Prerequisites

`.env` (loaded via `load_dotenv(find_dotenv())`, never hard-coded) must define:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`,
`BEDROCK_EMBEDDING_MODEL_ID`, `OPENSEARCH_COLLECTION` (+ `OPENSEARCH_AWS_ACCESS_KEY_ID`
/ `OPENSEARCH_AWS_SECRET_ACCESS_KEY` if the collection is in another account).

```bash
pip install -r requirements.txt opensearch-py
```

### One-time: build the KB index

```bash
cd multi-tool-server
python reindex.py            # chunks data/it_kb + data/policies (600w/100), embeds with Titan, bulk-loads novaops-kb
python reindex.py --force    # drop + rebuild unconditionally
```

`reindex.py` is a no-op if the indexed chunk count already matches `data/`. It
waits out OpenSearch Serverless's eventually-consistent drop/create and verifies
the final count, so repeated runs stay at exactly 27 chunks (no doubling/emptying).

### Run the server (terminal 1)

```bash
cd multi-tool-server/server
python server.py             # → http://127.0.0.1:9876/mcp
```

### Run the agent (terminal 2)

```bash
cd multi-tool-server
python agent.py "Any spare laptops in the US?"     # one-shot
python agent.py                                     # interactive REPL (exit / quit / q)
```

### Point the agent at a different MCP server

No code change — only the URL:

```bash
MCP_SERVER_URL=https://mcp.deepwiki.com/mcp python agent.py \
  "How does the Python MCP SDK implement the streamable-HTTP transport? repo modelcontextprotocol/python-sdk"
```

The agent connects, `list_tools()` now returns DeepWiki's tools
(`ask_question`, `read_wiki_contents`, `read_wiki_structure`), and the same loop
drives them. What stays the same: the transport, the `initialize` handshake,
`list_tools` / `call_tool`, the loop, the schema bridge, the model. What changes:
one environment variable.

### Run the eval

```bash
# server must be running
python multi-tool-server/eval/eval.py        # whole set
python multi-tool-server/eval/eval.py 3      # first 3 cases (quick smoke)
```

`eval/eval.py` runs `data/eval_tool_use.jsonl` through the same MCP loop as
`agent.py` (its own copy, not an import) and scores six metrics: tool **selection**
/ **necessity** / **argument correctness** (deterministic checks on the trajectory)
and answer **utilization** / **faithfulness** / **relevance** (LLM judges).

---

## 5. Where the MCP boundary sits

The boundary is the **tool interface**: a name, a JSON Schema, a result shape.

- **Above it** (agent): discovers tools, lets the model choose, runs calls, feeds
  results back. Provider-specific only in `mcp_tools_to_bedrock()`.
- **Below it** (server): owns *what the tools do* and *what they cost*. Task 3
  swapped `search_kb`'s entire backend from a local FAISS index to a network call
  to a managed vector store — different engine, different failure model, added
  latency, a Titan embedding call per query — and the agent never noticed: same
  tool name, same `{"article","score","text"}` shape, no re-list, no code change.

Adding a tool grows `server.py` only. Re-architecting a tool's backend touches only
that backend module. Talking to someone else's server changes only a URL.
