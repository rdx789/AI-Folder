# Agentic RAG Tool

An agentic RAG assistant (Amazon Bedrock Converse) that treats retrieval as a
tool the model decides whether to call — chit-chat gets answered directly,
policy/fact questions trigger one or more knowledge-base searches — rather
than retrieving on every turn. See `PROMPT.md` for the full spec.

## Layout

```
data/               one subfolder per knowledge base (each full of .md files)
data/<kb>/_description.txt   what the KB covers — read by registry.py, used for routing
data/<kb>/index/    built by ingest.py: index.faiss + chunks.json (chunk text + embedding-model id)
bedrock_client.py   single seam for every Bedrock call (converse + embed)
registry.py         auto-discovers knowledge bases by scanning data/
ingest.py           chunks + embeds + builds a FAISS index per knowledge base
search.py           loads an index and runs top-k retrieval
rag_tool.py         the agent loop + REPL (main())
```

Knowledge bases currently included: `handbook` (employee handbook) and
`manager_playbook` (manager-only guidance). Add a new one by dropping
`data/<name>/*.md` + `data/<name>/_description.txt` — no code changes needed.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in AWS credentials, region, BEDROCK_MODEL_ID, BEDROCK_EMBEDDING_MODEL_ID
```

Build (or rebuild) the indexes whenever `data/` content changes:

```bash
.venv/bin/python ingest.py
```

## Usage

```bash
.venv/bin/python rag_tool.py
```

Ask a question at the `>` prompt; type `quit` or `exit` to stop. Each
`search_knowledge_base` call the model makes is printed as a trace line
(`kb`, `query`) so you can see when it does — and doesn't — decide to search.

To use it from another app:

```python
from rag_tool import answer_question

answer_question("How many weeks of severance do employees get?")
```

## How retrieval is gated

`rag_tool.py` calls `converse()` with `tool_choice={"auto": {}}` in a loop
capped at `MAX_TURNS = 5`. The model decides per turn whether to call
`search_knowledge_base` (zero, one, or several times in the same turn); every
`toolUse` block gets a matching `toolResult` before the loop continues, and
the loop returns as soon as the model stops asking to search.
