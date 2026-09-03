# AI Folder

Course exercises building up Bedrock-based agents and RAG, lesson by lesson.
Each lesson lives in its own subfolder with its `PROMPT.md` (the original spec)
and `requirements.txt`; some also carry an `.env.example`.

## Lessons

| Folder | What it builds |
|---|---|
| [`03-add-tool-skill`](03-add-tool-skill) | A CLI smoke test for a customer-support use case: sends a representative support scenario to a Bedrock model through a tool-use loop and prints the answer. |
| [`04-create-agent-tool`](04-create-agent-tool) | A fuller customer-support agent with mock backend tools — order lookup, ticket status/creation, refund eligibility, and knowledge-base search — via Bedrock Converse. |
| [`05-coding-agents`](05-coding-agents) | A personal-finance-analyst agent that answers finance questions by querying a synthetic accounts/transactions dataset with dedicated tools, never by guessing numbers. |
| [`06-rag`](06-rag) | An agentic RAG assistant where retrieval is a tool the model decides whether to call — chit-chat is answered directly, policy questions trigger one or more knowledge-base searches, with knowledge bases auto-discovered from `data/`. |
| [`07-rag-chunking-study`](07-rag-chunking-study) | Lesson 6 homework: a research study comparing 8 chunking strategies (fixed-size, separator, sentence, semantic/LLM-based) over the same handbook corpus, each indexed and scored for right-source hit rate, answer correctness, and $ build cost — see [`RESULTS.md`](07-rag-chunking-study/RESULTS.md) for the comparison table and recommendation. |
| [`07-rag-advanced`](07-rag-advanced) | Lesson 7 homework: a production RAG pipeline — `retrieve → filter → rerank → answer` over an Amazon OpenSearch Serverless index (600w/100ov chunks, HNSW/faiss), with an access filter, a Nova subject planner, a Nova listwise reranker, and source-cited answers. Backed by an eval harness (Recall@k / MRR + 3 LLM judges over 17 labeled questions), an opt-in BM25+vector hybrid retriever, and a `rag-eval-loop` tuning skill. See [`PROMPT.md`](07-rag-advanced/PROMPT.md) for status and measured results. |
| [`08-mcp-2.0`](08-mcp-2.0) | Lesson 8 homework: extends the NovaOps MCP server from the lab — adds read tools (`check_asset_inventory`, `list_employee_tickets`), a write tool (`create_ticket`), and swaps the KB backend from local FAISS to an OpenSearch Serverless k-NN index, all with `agent.py` / `client.py` unchanged (the point: only the server grows). Build lives in [`multi-tool-server/`](08-mcp-2.0/multi-tool-server); backed by an LLM-judge eval over `data/eval_tool_use.jsonl`. |
| [`09-parallel-supervisor`](09-parallel-supervisor) | Lesson 9 homework (tasks 1–3): a LangGraph **supervisor** over MCP-tool specialists — a planner fans out to `knowledge` / `records` / `eligibility` sub-agents in parallel (`Send` + `operator.add`), a reflection step runs a sharper second round when the findings fall short (capped at 2), and the one write action (`create_access_request`) sits behind a durable human approval gate: `AsyncSqliteSaver` checkpoint, one-process inline y/n, with `resume` to recover a run killed at the prompt. See [`PROMPT.md`](09-parallel-supervisor/PROMPT.md). |
| [`09-router-agent`](09-router-agent) | Lesson 9 homework (task 4): the opposite pattern — a top-level **router** whose `classify` node labels a request `policy_qa` / `it_ticket` / `access_request` / `lookup` / `general` and a conditional edge sends it down exactly one path, each a focused `create_agent` over just that category's tools (or a plain node). Clean single-path routing — no fan-out, no merging. See [`PROMPT.md`](09-router-agent/PROMPT.md). |

## Common setup pattern

Each lesson is self-contained:

```bash
cd <lesson-folder>
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # where present; otherwise create .env with
                       # AWS credentials, region, BEDROCK_MODEL_ID, etc.
```

See each lesson's own `PROMPT.md` for what it builds and how to run it.
