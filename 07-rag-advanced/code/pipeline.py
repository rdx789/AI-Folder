"""retrieve -> filter -> rerank -> answer, the filter+rerank MIX:
  1. retrieve  — wide k-NN over the shared index, audience-filtered at the query level.
  2. filter    — subject planner narrows the wide set to on-topic chunks (soft: falls
                 back to the full wide set if narrowing would empty it), then a
                 recency-aware sort breaks ties among equally relevant chunks.
  3. rerank    — Nova listwise reranker scores the (already narrowed) candidates and
                 keeps a small final top-k.
  4. answer    — Nova answers using only the reranked chunks, citing the source file
                 for every claim.
"""
import json

from env_check import validate_env

REFUSAL = "I don't have that information in the context."

validate_env()

from client import INDEX_NAME, TOP_K, bedrock, embed_text, opensearch_client  # noqa: E402
from subjects import SUBJECTS, tag_subjects, _model_id  # noqa: E402

RETRIEVE_WIDE = 20


def access_filter(audience: str) -> list[str]:
    """Which document `audience` values a caller may see."""
    if audience == "manager":
        return ["all", "manager"]
    return ["all"]


def retrieve(client, query: str, audience: str, wide_k: int = RETRIEVE_WIDE) -> list[dict]:
    vector = embed_text(query)
    body = {
        "size": wide_k,
        "query": {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": vector, "k": wide_k}}}],
                "filter": [{"terms": {"audience": access_filter(audience)}}],
            }
        },
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return [hit["_source"] | {"_score": hit["_score"]} for hit in resp["hits"]["hits"]]


def filter_by_subject_and_recency(candidates: list[dict], planned_subjects: list[str]) -> list[dict]:
    if planned_subjects:
        narrowed = [c for c in candidates if set(c.get("subjects", [])) & set(planned_subjects)]
        if narrowed:
            candidates = narrowed
    return sorted(candidates, key=lambda c: (c.get("last_updated") or ""), reverse=True)


_RERANK_TOOL = {
    "toolSpec": {
        "name": "submit_ranking",
        "description": "Submit the chunk indices ordered from most to least relevant.",
        "inputSchema": {"json": {
            "type": "object", "additionalProperties": False, "required": ["ranked_indices"],
            "properties": {
                "ranked_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Chunk indices (0-based), most relevant first.",
                },
            },
        }},
    }
}


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
    """One listwise Nova call: sees all candidates together and ranks them, rather
    than scoring each chunk independently (which can't compare chunks to each other)."""
    if not candidates:
        return []
    numbered = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    resp = bedrock.converse(
        modelId=_model_id(),
        system=[{"text": (
            "Rank the numbered CHUNKS by relevance to the QUESTION, most relevant first. "
            "Return ALL chunk indices via the tool."
        )}],
        messages=[{"role": "user", "content": [{"text": f"### QUESTION\n{query}\n\n### CHUNKS\n{numbered}"}]}],
        toolConfig={"tools": [_RERANK_TOOL], "toolChoice": {"tool": {"name": "submit_ranking"}}},
        inferenceConfig={"temperature": 0.0},
    )
    ranked_indices = []
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            ranked_indices = block["toolUse"]["input"].get("ranked_indices", [])
    ranked = [candidates[i] for i in ranked_indices if 0 <= i < len(candidates)]
    return ranked[:top_k] if ranked else candidates[:top_k]


def answer(query: str, contexts: list[dict]) -> str:
    if not contexts:
        return REFUSAL
    numbered = "\n\n".join(f"[{i + 1}] (source: {c['key']}) {c['text']}" for i, c in enumerate(contexts))
    resp = bedrock.converse(
        modelId=_model_id(),
        system=[{"text": (
            "Answer the QUESTION using ONLY the CONTEXT chunks below. Every claim you make "
            "must cite the source file it came from, like (source: handbook/severance.md). "
            "If the context does not contain the answer, respond with EXACTLY this sentence "
            f"and nothing else: \"{REFUSAL}\" Do not guess, "
            "and do not partially answer from outside knowledge."
        )}],
        messages=[{"role": "user", "content": [{"text": f"### QUESTION\n{query}\n\n### CONTEXT\n{numbered}"}]}],
        inferenceConfig={"temperature": 0.0},
    )
    return resp["output"]["message"]["content"][0]["text"]


def run(query: str, audience: str, top_k: int = TOP_K, wide_k: int = RETRIEVE_WIDE,
        use_hybrid: bool = False, verbose: bool = True) -> dict:
    client = opensearch_client()
    planned_subjects = tag_subjects(query)
    if use_hybrid:
        from hybrid_search import hybrid_search
        wide = hybrid_search(client, query, audience, wide_k)
    else:
        wide = retrieve(client, query, audience, wide_k)
    narrowed = filter_by_subject_and_recency(wide, planned_subjects)
    top = rerank(query, narrowed, top_k)
    final_answer = answer(query, top)

    result = {
        "query": query,
        "audience": audience,
        "access_filter": access_filter(audience),
        "planned_subjects": planned_subjects,
        "reranked_top_k": [{"key": c["key"], "text": c["text"][:200]} for c in top],
        "top_keys": [c["key"] for c in top],       # ordered, for Recall@k / MRR
        "contexts": [c["text"] for c in top],      # full text, for the LLM judges
        "answer": final_answer,
    }
    if verbose:
        print(f"\n=== {audience.upper()} QUERY ===\n{query}")
        print(f"access_filter: {result['access_filter']}")
        print(f"planned_subjects: {result['planned_subjects']}")
        print("reranked_top_k:")
        for c in result["reranked_top_k"]:
            print(f"  - {c['key']}: {c['text']}...")
        print(f"\nANSWER:\n{final_answer}")
    return result


if __name__ == "__main__":
    run("If I've worked here for 3 years and get laid off, how much severance do I get?", "employee")
    run("What steps do I need to take before terminating a report for underperformance?", "manager")
