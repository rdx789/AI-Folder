The build lives in `multi-tool-server/`. Across all four tasks `agent.py` and
`client.py` are unchanged — only the server, the retrieval backend, or the server
URL moves.

1. Add two read-only MCP tools to `server/server.py`, each backed by a
   function in `server/db.py`, in the shape of the existing `check_software_subscription`:
   `check_asset_inventory(asset_type, location)` returning `status='available'` rows
   from the assets table (both filters optional, partial, case-insensitive), and
   `list_employee_tickets(employee_id)` returning that employee's rows from the
   tickets table, newest first. Restart the server and confirm the agent calls the
   new tool for both "Any spare laptops in the US?" (correctly: none — the one spare
   laptop is in Israel) and "What tickets does E010 have open?", with agent.py
   unchanged.

2. Add a write tool `create_ticket(employee_id, subject, description, category)` in
   `server/db.py` + `server/server.py`, in the shape of the existing
   `create_access_request`: check the employee exists, insert into the tickets
   table with a generated id `TCK-0001`, status `open`, `assigned_team` derived
   from the category (Security→Security, HR/People→People, Facilities→Facilities,
   else→IT), and `subcategory`/`priority` defaulted to `General`/`medium`; return
   the created row. No approval logic — that's a later lesson. Show the agent
   filing a ticket for "E010 can't connect to the VPN", with agent.py unchanged.

3. Rewrite `server/rag.py` so `search_kb(query)` runs a k-NN vector search
   against the OpenSearch Serverless collection (`OPENSEARCH_COLLECTION` from
   `.env`, clients in `kb_client.py`, vector field `embedding`) instead of a
   local FAISS index — keeping the exact same function signature and return shape
   (`list[{"article", "score", "text"}]`). `reindex.py` builds the
   `novaops-kb` index from `data/it_kb/` + `data/policies/` (600-word / 100-overlap
   chunks, Titan embeddings). Don't change `server/server.py`'s tool or agent.py.
   Ask a knowledge-base question and confirm the answer now comes from OpenSearch.
   Explain what this shows about where the MCP boundary sits.

4. Run `agent.py` against the public DeepWiki MCP server instead of the local
   one — set `MCP_SERVER_URL=https://mcp.deepwiki.com/mcp` (no auth). Ask "How does
   the Python MCP SDK implement the streamable-HTTP transport?" about the repo
   `modelcontextprotocol/python-sdk`. Show the tools it discovered
   (`ask_question`, `read_wiki_contents`, `read_wiki_structure`), and confirm no
   code changed — only the server URL. Explain what had to change to talk to a
   server I didn't write, and what didn't.