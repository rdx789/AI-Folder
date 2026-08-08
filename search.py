"""Retrieval against a knowledge base's pre-built FAISS index (see ingest.py)."""
import json

import faiss
import numpy as np

from bedrock_client import EMBEDDING_MODEL_ID, embed
from registry import KnowledgeBase

_index_cache: dict[str, tuple] = {}  # kb_name -> (faiss_index, chunk_records)


def _load(kb: KnowledgeBase):
    if kb.name in _index_cache:
        return _index_cache[kb.name]

    if not kb.faiss_path.is_file() or not kb.chunks_path.is_file():
        raise RuntimeError(
            f"No index found for knowledge base '{kb.name}' at {kb.index_dir}. "
            "Run `python ingest.py` first."
        )

    index = faiss.read_index(str(kb.faiss_path))
    metadata = json.loads(kb.chunks_path.read_text(encoding="utf-8"))
    if metadata["embedding_model"] != EMBEDDING_MODEL_ID:
        raise RuntimeError(
            f"Index for '{kb.name}' was built with '{metadata['embedding_model']}', "
            f"but BEDROCK_EMBEDDING_MODEL_ID is '{EMBEDDING_MODEL_ID}'. Re-run `python ingest.py`."
        )
    records = metadata["chunks"]
    _index_cache[kb.name] = (index, records)
    return index, records


def search(kb: KnowledgeBase, query: str, top_k: int = 4) -> list[dict]:
    index, records = _load(kb)
    query_vector = np.array([embed(query)], dtype="float32")
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))
    return [
        {**records[i], "score": float(score)}
        for score, i in zip(scores[0], indices[0])
        if i != -1
    ]