"""hybrid_search: BM25 (exact-term) + k-NN (semantic), fused with reciprocal-rank
fusion (RRF). Fixes what pure vectors fumble on exact terms — an error code, a
policy name, a specific dollar figure — since BM25 rewards literal term overlap
that a paraphrase-tolerant embedding can dilute.

BM25 scores and cosine-ish k-NN scores live on different, incomparable scales, so
we never mix the raw scores. RRF sidesteps that entirely: it only looks at each
list's RANK order, so no normalization guesswork.
"""
from client import INDEX_NAME, embed_text

RRF_K = 60  # standard RRF damping constant — de-weights rank differences deep in the list


def _bm25_search(client, query: str, audience_values: list[str], top_n: int) -> list[dict]:
    body = {
        "size": top_n,
        "query": {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": [{"terms": {"audience": audience_values}}],
            }
        },
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return [hit["_source"] | {"_score": hit["_score"]} for hit in resp["hits"]["hits"]]


def _knn_search(client, query: str, audience_values: list[str], top_n: int) -> list[dict]:
    vector = embed_text(query)
    body = {
        "size": top_n,
        "query": {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": vector, "k": top_n}}}],
                "filter": [{"terms": {"audience": audience_values}}],
            }
        },
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return [hit["_source"] | {"_score": hit["_score"]} for hit in resp["hits"]["hits"]]


def hybrid_search(client, query: str, audience: str, top_k: int, top_n: int = 20) -> list[dict]:
    """Runs BM25 `match` on `text` and k-NN on `embedding`, each top_n, with the
    same audience filter on both, then fuses by reciprocal-rank fusion:
    score(doc) = sum over lists containing doc of 1 / (RRF_K + rank_in_that_list).
    Returns the top_k fused documents, highest score first."""
    from pipeline import access_filter  # local import avoids a circular import at module load

    audience_values = access_filter(audience)
    bm25_hits = _bm25_search(client, query, audience_values, top_n)
    knn_hits = _knn_search(client, query, audience_values, top_n)

    rrf_scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for ranked_list in (bm25_hits, knn_hits):
        for rank, doc in enumerate(ranked_list, start=1):
            chunk_id = doc["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            docs.setdefault(chunk_id, doc)

    fused_order = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [docs[cid] | {"_rrf_score": rrf_scores[cid]} for cid in fused_order[:top_k]]
