"""Chunk every knowledge base's .md files and build a per-KB FAISS index.

Run manually whenever data/<kb>/ content changes:
    python ingest.py
"""
import json

import faiss
import numpy as np

from bedrock_client import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID, embed
from registry import get_registry

CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150


def chunk_text(text: str) -> list[str]:
    """Split markdown into paragraph-aware chunks of ~CHUNK_CHARS with overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > CHUNK_CHARS:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_index(kb):
    md_files = sorted(kb.source_dir.glob("*.md"))
    if not md_files:
        print(f"  [{kb.name}] no .md files found, skipping")
        return

    records = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            records.append({"text": chunk, "source": path.name})

    print(f"  [{kb.name}] embedding {len(records)} chunks from {len(md_files)} files...")
    vectors = np.array([embed(r["text"]) for r in records], dtype="float32")

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(vectors)

    kb.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(kb.faiss_path))
    metadata = {"embedding_model": EMBEDDING_MODEL_ID, "chunks": records}
    kb.chunks_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{kb.name}] wrote index with {index.ntotal} vectors -> {kb.index_dir}")


def main():
    registry = get_registry()
    print(f"Found {len(registry)} knowledge base(s): {', '.join(registry)}")
    for kb in registry.values():
        build_index(kb)


if __name__ == "__main__":
    main()