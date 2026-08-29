# NovaOps MCP assistant

An MCP server exposing nine NovaOps tools, and a Bedrock agent that discovers and
calls them over the wire. The point of every task below: **the agent never
changes.** `agent.py` and `client.py` are identical across all four; only the
*server* grows, the *retrieval backend* swaps, or the *server URL* changes.

```
multi-tool-server/
  agent.py       # the MCP tool-use loop — never edited
  client.py      # Bedrock Converse + MCP→Bedrock tool-schema bridge — never edited
  kb_client.py   # OpenSearch Serverless + Titan embedding clients
  reindex.py     # builds the novaops-kb search index from data/
  server/
    server.py    # the nine @mcp.tool() definitions
    db.py        # SQLite engine behind the database tools
    rag.py       # search_kb() — k-NN over OpenSearch
  eval/          # provided scorers — untouched
```

## Setup

Config is read only from `../.env` via `load_dotenv(find_dotenv())` — nothing is
hard-coded. Required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
`BEDROCK_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID`, `OPENSEARCH_COLLECTION`
(+ `OPENSEARCH_AWS_ACCESS_KEY_ID` / `OPENSEARCH_AWS_SECRET_ACCESS_KEY` if the
collection lives in a different account). `pip install -r ../requirements.txt`
plus `opensearch-py`.

```bash
cd multi-tool-server && python reindex.py       # load data/ into the novaops-kb index
cd multi-tool-server/server && python server.py  # → http://127.0.0.1:9876/mcp
# in another shell:
cd multi-tool-server && python agent.py "..."
```

`reindex.py` only rebuilds when the chunk count no longer matches what's indexed
(or the index is missing/empty); `--force` rebuilds unconditionally. It waits out
OpenSearch Serverless's eventually-consistent drop/create and verifies the final
count, so repeated runs settle at exactly 27 chunks instead of doubling or emptying.

---

## 1 — Two read-only database tools

`check_asset_inventory(asset_type, location)` and `list_employee_tickets(employee_id)`
in `db.py`, each wrapped by an `@mcp.tool()` in `server.py`, built in the same shape
as `check_software_subscription`: a plain parameterised SQL read returning
`list[dict]`. `check_asset_inventory` returns only `status='available'` rows,
narrowed by partial case-insensitive matches on type and location.

```
$ python agent.py "Any spare laptops in the US?"
Connected. Tools: list_policies, get_policy, search_knowledge_base, get_employee,
  check_software_subscription, check_asset_inventory, list_employee_tickets,
  create_ticket, create_access_request

  → MCP tool: check_asset_inventory({'asset_type': 'Laptop', 'location': 'US'})

Assistant:
There are currently no spare laptops available in the US inventory. ...
```

```
$ python agent.py "What tickets does E010 have open?"
  → MCP tool: list_employee_tickets({'employee_id': 'E010'})

Assistant:
E010 currently has one open ticket:
**Ticket T001** — Access (Webex license), Priority High, Status Open,
"Webex says account is not licensed" ...
```

The agent discovered each new tool and picked the right one per question, no code
change. The laptop answer is correct: the one unassigned laptop in the seed data
(`A005`, MacBook Pro 14) is in Israel, so there genuinely are none in the US.

## 2 — A write tool: IT support tickets

`create_ticket(employee_id, subject, description, category)` in `db.py` + `server.py`,
same shape as `create_access_request`: confirm the employee exists, mint a
human-readable id, insert, return the created row.

- id: `TCK-0001`, `TCK-0002`, … kept in a sequence of its own
- `status`: `open`
- `assigned_team`: derived from `category` — `Security`→Security, `HR`/`People`→People,
  `Facilities`→Facilities, everything else→IT
- `subcategory` / `priority`: default to `General` / `medium` (not in the signature)
- **no approval logic** — an open ticket just enters the queue

```
$ python agent.py "File a support ticket for E010: subject 'Cannot connect to the VPN'.
   Description: since this morning the GlobalProtect client fails to connect from home
   with an authentication error; Okta login itself works. Category Network. Create it now."

  → MCP tool: create_ticket({'employee_id': 'E010', 'subject': 'Cannot connect to the VPN',
      'description': 'since this morning the GlobalProtect client fails ...', 'category': 'Network'})

Assistant:
Your support ticket has been filed successfully:
- Ticket ID: TCK-0001
- Status: Open (assigned to IT)
...
```

Writes live in the in-memory SQLite DB for the life of the server process, same as
`create_access_request`.

## 3 — `search_kb()` on OpenSearch instead of a local index

`server/rag.py` rewritten. **Unchanged:** the signature `search_kb(query) -> list[dict]`
and the return shape `{"article", "score", "text"}`. **Changed:** the body — it now
issues a k-NN query against the managed `novaops-kb` OpenSearch Serverless
collection (`kb_client.opensearch_client()` + Titan `embed_text()`), targeting the
`embedding` vector field. `warm_index()` went from building a local index to
confirming the managed one is reachable and reporting its doc count.

`server/server.py`'s `search_knowledge_base` tool and `agent.py` were not touched.

```
$ python agent.py "The knowledge base — how do I reset my MFA / get a new device enrolled in Okta?"
  → MCP tool: search_knowledge_base({'query': 'reset MFA Okta new device enrollment'})

Assistant:
Here are the most relevant knowledge base articles ...
1. MFA Reset (IT-KB-103) — Resetting your Okta MFA factor and re-enrolling immediately ...
2. Okta Device Change (IT-KB-104) — Changing or adding a device to Okta ...
3. Security and MFA Policy — All accounts must use MFA via Okta, no exceptions ...
```

The answer quotes `IT-KB-103` / `IT-KB-104` verbatim from `data/it_kb/mfa_reset.md`
and `okta_device_change.md` — retrieved from OpenSearch. Server startup prints
`KB index reachable — 27 chunks searchable.`

### Where the MCP boundary sits

The boundary is **the tool interface** — a name, a JSON schema, a result shape.
Everything below it is the server's private business. Replacing an in-process local
vector index with a network call to a managed store — different engine, different
failure model, added latency, possibly a different AWS account paying for it — was
invisible to the agent: it never re-listed tools, never changed a call. Capability
*and cost* also sit server-side: the embedding call that turns the query into a
vector runs in `rag.py`, on the server, not in the agent. The whole backend of a
tool can be re-architected without redeploying, retesting, or restarting anything
on the client.

## Eval

`python eval/eval.py` (with the server running) scores the agent's tool use over
`data/eval_tool_use.jsonl` — three deterministic trajectory checks and three
LLM-judge outcome metrics.

| group  | Selection | Necessity | Arg-correct | Utilization | Faithful | Relevance |
|--------|-----------|-----------|-------------|-------------|----------|-----------|
| ALL    | 0.92      | 0.96      | 1.00        | 1.00        | 1.00     | 1.00      |
| easy   | 1.00      | 1.00      | 1.00        | 1.00        | 1.00     | 1.00      |
| medium | 1.00      | 0.90      | 1.00        | 1.00        | 1.00     | 1.00      |
| hard   | 0.67      | 1.00      | 1.00        | 1.00        | 1.00     | 1.00      |

The new tools introduced no regression — arg-correctness is 1.00 (the
disambiguation cases still pass) and every tool-grounded answer is faithful and
relevant. The one miss is `VAGUE_CANT_LOGIN` ("I can't get into one of my tools,
what do I do?"): the dataset expects `search_knowledge_base`, but the agent asked
the user to clarify instead of calling any tool — a `nova-2-lite` judgment call on
a deliberately under-specified prompt, not a server/tool problem. That single case
pulls hard-group Selection to 0.67 and ALL to 0.92.

## 4 — The same agent against a server we didn't write

No code change — just the URL:

```
$ MCP_SERVER_URL=https://mcp.deepwiki.com/mcp python agent.py \
    "How does the Python MCP SDK implement the streamable-HTTP transport?
     The repo is modelcontextprotocol/python-sdk."

Connected. Tools: ask_question, read_wiki_contents, read_wiki_structure

  → MCP tool: ask_question({'question': 'How does the Python MCP SDK implement the
      streamable-HTTP transport?', 'repoName': 'modelcontextprotocol/python-sdk'})

Assistant:
The Python MCP SDK implements the streamable-HTTP transport through a client-side
`StreamableHTTPTransport` class and a server-side `StreamableHTTPServerTransport`
class, managed by a `StreamableHTTPSessionManager`. ...
```

Tools discovered on the third-party server: **`ask_question`, `read_wiki_contents`,
`read_wiki_structure`** — a completely different set, discovered at connect time and
translated to Bedrock's format by the same bridge in `client.py`.

### What had to change vs. what didn't

**Changed:** one environment variable, `MCP_SERVER_URL`. No source edit — `git diff`
is empty.

**Didn't change:** the transport (`streamable-http` — DeepWiki speaks the same MCP
2.0 wire protocol our server does), the `initialize` handshake, `list_tools`,
`call_tool`, the tool-use loop, the schema bridge, the model. Because DeepWiki
implements the spec, "a server someone else wrote" and "a server I wrote" are the
same client code path. The only inherently server-specific things — tool names,
schemas, behavior — are exactly what the agent *discovers* rather than hard-codes.
(DeepWiki needs no auth; one that did would add a token to the `Client(...)` call —
still config, not logic.)