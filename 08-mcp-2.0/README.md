# Homework — MCP

Extensions of the NovaOps MCP server you built in the lab. Each one grows the
server (the agent barely changes — that's the point) and produces a piece you'll
reuse in the final project. The prompt hint helps you drive a coding agent through
it; copy it, or write your own.

All of these build on **Exercise 3** (`03-multi-tool-server/`). Start its server in one
terminal and iterate from another.

## 1. Operational lookup tools (read the database)

**Goal.** Grow the DB **read-tool suite** so the agent can answer more operational
questions. The lab already ships `get_employee` and `check_software_subscription`;
you'll add two more in the same shape. You'll reuse these as the **operational read
tools** in the final project.

**What to build.** Add two read-only tools to `03-multi-tool-server/server/server.py`, backed
by new functions in `server/db.py`, following the exact pattern of the shipped
`check_software_subscription`: `check_asset_inventory(asset_type, location)`
(available rows from `assets`) and `list_employee_tickets(employee_id)` (that
person's rows from `tickets`). The data is already in `data/database/seed.sql`.
Confirm the agent picks the right tool for "Any spare laptops in the US?" and
"What tickets does E010 have open?" — and that you never touched `agent.py`.

**Prompt hint.**
> "In `03-multi-tool-server/`, add two read-only MCP tools to `server/server.py`, each backed
> by a function in `server/db.py`, copying the pattern of the existing
> `check_software_subscription` tool: `check_asset_inventory(asset_type, location)`
> returning available rows from `assets`, and `list_employee_tickets(employee_id)`
> returning that employee's rows from `tickets`. Then restart the server and ask the
> agent 'Any spare laptops in the US?' — show me it calls the new tool, with
> `agent.py` unchanged."

## 2. A second write tool — IT support tickets

**Goal.** The lab ships one write tool (`create_access_request`). Add a second one
in the same shape, for a different operational domain: IT support tickets. This is
the **mock operational tools** layer of the final project. Note what
you are *not* doing — deciding or approving the action. `create_access_request`
already models the boundary (file it `pending_approval`, let a human dispose); the
**approval gate itself is Lesson 9's job**, so keep that logic out of the tool.

**What to build.** Add `create_ticket(employee_id, subject, description, category)`
to `server/db.py` + `server/server.py`, following the shipped `create_access_request`: validate the
employee exists, insert a row into the `tickets` table with a generated id
(`TCK-0001`…) and a sensible default `status`/`assigned_team` (derive the team from
`category`), and return the created record. Confirm the agent files a ticket for
"E010 can't connect to the VPN" — and that `agent.py` is still untouched.

**Prompt hint.**
> "Extend `03-multi-tool-server/` with a write tool `create_ticket(employee_id,
> subject, description, category)` in `server/db.py` + `server/server.py`, copying the pattern of
> the existing `create_access_request` tool: check the employee exists, insert into
> the `tickets` table with a generated id like `TCK-0001`, an 'open' status, and an
> `assigned_team` derived from the category, then return the row. Don't add any
> approval logic — that's a later lesson. Show me the agent filing a ticket for 'E010
> can't connect to the VPN', with `agent.py` unchanged."

## 3. Serve Lesson 7's real retrieval through the same tool

**Goal.** Replace the lab's local FAISS index with the **OpenSearch retrieval you
built in Lesson 7**, behind the *same* `search_knowledge_base` tool. This makes the
MCP server expose production-grade retrieval — the **retrieval core of the final
project**, now reachable by any MCP client.

**What to build.** Rewrite `03-multi-tool-server/server/rag.py` so `search_kb` queries your
Lesson 7 OpenSearch Serverless collection (vector search, optionally with the
metadata filtering/reranking you added) instead of the in-memory FAISS index. The
tool's signature and the server stay the same — only the engine changes. Prove that
the agent's behavior and code are untouched while the retrieval backend is now the
managed store.

**Prompt hint.**
> "Rewrite `03-multi-tool-server/server/rag.py` so `search_kb(query)` runs against my Lesson 7
> OpenSearch Serverless collection (reuse that lab's `client.py` / search code and
> `OPENSEARCH_COLLECTION` from `.env`) instead of the local FAISS index — keeping
> the exact same function signature and return shape. Don't change `server/server.py`'s tool
> or `agent.py`. Then ask a knowledge-base question and confirm the answer now comes
> from OpenSearch. Explain what this shows about where the MCP boundary sits."

## 4. *(Optional)* Point your agent at a remote server you didn't write

**Goal.** Experience the other half of MCP — **using tools you didn't write**, hosted
by **someone else on the internet**. This is off-project (an ecosystem exercise, not a
NovaOps piece), so it's optional: it builds intuition for why a *standard* protocol
matters. The payoff: connecting to a stranger's server takes **zero code changes** —
the same discovery + loop you already have just works against a new URL.

**What to build.** Point `02-agent/agent.py` at the public **DeepWiki** MCP server —
`https://mcp.deepwiki.com/mcp`, no signup, no API key — by setting `MCP_SERVER_URL`.
It speaks the **same streamable-HTTP transport** as your own server, so nothing in
your code changes: the agent connects, discovers DeepWiki's tools (`read_wiki_structure`,
`read_wiki_contents`, `ask_question`), and answers questions about any public GitHub
repo. Ask it something about a real project and watch your unchanged loop drive a
server you've never seen. (If the endpoint is ever down, the official reference servers
at `github.com/modelcontextprotocol/servers` are a local stdio fallback — but that
needs a transport swap, which is exactly the friction DeepWiki avoids.)

**Prompt hint.**
> "Run my `02-agent/agent.py` against the public DeepWiki MCP server instead of the
> NovaOps one — set `MCP_SERVER_URL=https://mcp.deepwiki.com/mcp` (no auth needed).
> Then ask it 'How does the Python MCP SDK implement the streamable-HTTP transport?'
> about the repo `modelcontextprotocol/python-sdk`. Show me the tools it discovered,
> and confirm I changed no code — only the server URL. Explain what had to change to
> talk to a server I didn't write, and what didn't."
