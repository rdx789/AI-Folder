# AI Folder

Course exercises building up Bedrock-based agents and RAG, lesson by lesson.
Each lesson lives in its own subfolder with its own `README.md`, `PROMPT.md`
(the original spec), `.env.example`, and `requirements.txt`.

## Lessons

| Folder | What it builds |
|---|---|
| [`03-add-tool-skill`](03-add-tool-skill) | A CLI smoke test for a customer-support use case: sends a representative support scenario to a Bedrock model through a tool-use loop and prints the answer. |
| [`04-create-agent-tool`](04-create-agent-tool) | A fuller customer-support agent with mock backend tools — order lookup, ticket status/creation, refund eligibility, and knowledge-base search — via Bedrock Converse. |
| [`05-coding-agents`](05-coding-agents) | A personal-finance-analyst agent that answers finance questions by querying a synthetic accounts/transactions dataset with dedicated tools, never by guessing numbers. |
| [`06-rag`](06-rag) | An agentic RAG assistant where retrieval is a tool the model decides whether to call — chit-chat is answered directly, policy questions trigger one or more knowledge-base searches, with knowledge bases auto-discovered from `data/`. |
| [`rag-chunking-study`](rag-chunking-study) | Lesson 6 homework: a research study comparing 8 chunking strategies (fixed-size, separator, sentence, semantic/LLM-based) over the same handbook corpus, each indexed and scored for right-source hit rate, answer correctness, and $ build cost — see [`RESULTS.md`](rag-chunking-study/RESULTS.md) for the comparison table and recommendation. |
| [`07-rag-advanced`](07-rag-advanced) | Lesson 7 homework: a production RAG pipeline — `retrieve → filter → rerank → answer` over an Amazon OpenSearch Serverless index (600w/100ov chunks, HNSW/faiss), with an access filter, a Nova subject planner, a Nova listwise reranker, and source-cited answers. Backed by an eval harness (Recall@k / MRR + 3 LLM judges over 17 labeled questions), an opt-in BM25+vector hybrid retriever, and a `rag-eval-loop` tuning skill. See [`PROMPT.md`](07-rag-advanced/PROMPT.md) for status and measured results. |

## Common setup pattern

Each lesson is self-contained:

```bash
cd <lesson-folder>
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in AWS credentials, region, BEDROCK_MODEL_ID, etc.
```

See each lesson's own `README.md` for how to run it.
