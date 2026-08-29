"""
The knowledge-base tool's engine (server side) — backed by OpenSearch Serverless.

search_kb(query) runs a k-NN vector search against the managed 'novaops-kb'
collection and returns a list of {"article", "score", "text"} dicts. That
signature and return shape are the contract server.py's search_knowledge_base tool
depends on; the agent only ever sees the tool. Swapping the search backend — a
local index, a managed store, anything — is invisible on the other side of the MCP
boundary as long as this function's shape holds.

The index stores its vector in a field named `embedding` (1024-dim, faiss / HNSW /
innerproduct) alongside `text` + `source` metadata; reindex.py builds it from
data/it_kb/ and data/policies/.
"""

import sys
from pathlib import Path

# server.py runs from this server/ directory; kb_client.py lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb_client import INDEX_NAME, TOP_K, embed_text, opensearch_client  # noqa: E402

VECTOR_FIELD = "embedding"

# One client for the life of the server process. Lazily built so importing this
# module never does I/O.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = opensearch_client()
    return _client


def warm_index() -> int:
    """Confirm the KB index is reachable and return how many chunks are searchable.
    Called at server startup so a broken backend fails loudly then, not on the
    first question."""
    try:
        client = _get_client()
        if not client.indices.exists(index=INDEX_NAME):
            raise RuntimeError(
                f"Index '{INDEX_NAME}' not found in the collection — ingest the KB first."
            )
        return client.count(index=INDEX_NAME).get("count", 0)
    except Exception as e:  # noqa: BLE001 — startup check: report loudly and re-raise
        raise RuntimeError(f"Cannot reach the OpenSearch KB index: {e}") from e


def search_kb(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k KB chunks most similar to `query`, best first, as a list of
    {"article", "score", "text"} dicts. No access filter — this internal assistant
    sees the whole KB."""
    try:
        body = {
            "size": k,
            "query": {"knn": {VECTOR_FIELD: {"vector": embed_text(query), "k": k}}},
            "_source": ["text", "source", "key"],
        }
        hits = _get_client().search(index=INDEX_NAME, body=body)["hits"]["hits"]
    except Exception as e:  # noqa: BLE001 — a flaky managed-store call shouldn't crash the server
        return [{"article": "error", "score": 0.0, "text": f"Knowledge-base search failed: {e}"}]

    results = []
    for hit in hits:
        src = hit.get("_source", {})
        results.append(
            {
                "article": src.get("source") or src.get("key") or hit.get("_id", ""),
                "score": round(float(hit.get("_score", 0.0)), 3),
                "text": src.get("text", ""),
            }
        )
    return results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I get VPN access?"
    print(f"query: {q}\n")
    for r in search_kb(q):
        print(f"[{r['score']}] {r['article']}\n{r['text'][:200]}...\n")
