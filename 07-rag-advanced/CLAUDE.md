<!-- Rename this file to CLAUDE.md to activate it, then build from the lesson's PROMPT.md (in the reference repo's SDD/ folder). You're building a filtered + reranked RAG pipeline over an index you tag yourself, and an eval that measures the mix. -->

# CLAUDE.md — conventions for this rebuild

- Only `client.py` (OpenSearch + Bedrock wiring) and `judges.py` (the three eval judges) are PROVIDED — import from them, don't rewrite them. Everything else is yours to build: the subject vocabulary + tagger, the index mapping, the ingest, the filters, the planner, the reranker, and the eval.
- **First, validate the `.env`** (loaded via `load_dotenv(find_dotenv())`): confirm every var below is set, and if any is missing, STOP and ask me to add it — never guess, default, or hard-code a value. Vars: `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (Bedrock access), `AWS_REGION` (the collection's region, e.g. `us-east-1`), `BEDROCK_MODEL_ID` (`us.amazon.nova-2-lite-v1:0`), `BEDROCK_EMBEDDING_MODEL_ID` (`amazon.titan-embed-text-v2:0`), and `OPENSEARCH_COLLECTION` (the live collection's name — the endpoint resolves from it). This slate ships no `.env.example`; if the `.env` is missing, ask me to bring the one I used for the rest of this lesson.
- The OpenSearch collection is already provisioned; you build the index INTO it. Run your `create_index.py` then `ingest.py` once to populate `INDEX_NAME` before you query it.
- Call models with boto3 Bedrock Converse; keep each model call in one small function so a provider swap is easy. For structured output (subject tags, rerank scores), a forced tool call beats parsing prose.
- Config from env: `.env` via `load_dotenv(find_dotenv())`; model ID + region from env vars, never hard-coded.
- Comment only the non-obvious AI/SDK bits — skip the obvious.
