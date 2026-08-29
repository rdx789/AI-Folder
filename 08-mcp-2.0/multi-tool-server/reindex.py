"""Build the OpenSearch Serverless 'novaops-kb' index from data/it_kb/ + data/policies/.

This is what search_kb() (server/rag.py) searches. Each .md file is chunked at
600 words / 100 overlap, embedded with Titan, and bulk-loaded with its filename and
corpus as metadata.

    python reindex.py          # rebuild only if something changed (see below)
    python reindex.py --force  # drop + rebuild unconditionally

Re-embedding every chunk is real Bedrock spend, so by default this only rebuilds
when it would actually differ: the index is missing/empty, or the chunk count no
longer matches what's indexed (the cheap proxy for "source docs or chunk config
changed"). Otherwise it's a no-op.
"""
import sys
import time
from pathlib import Path

from opensearchpy import helpers

from kb_client import EMBED_DIM, INDEX_NAME, embed_text, opensearch_client

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPORA = ("it_kb", "policies")
CHUNK_WORDS = 600
OVERLAP_WORDS = 100

MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "innerproduct",
                    "parameters": {"ef_construction": 512, "m": 16},
                },
            },
            "text": {"type": "text"},
            "source": {"type": "keyword"},   # filename, e.g. 'vpn_access.md'
            "key": {"type": "keyword"},      # 'it_kb/vpn_access.md' — unique across corpora
            "corpus": {"type": "keyword"},   # 'it_kb' | 'policies'
            "audience": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
        }
    },
}


def _wait(predicate, what: str, timeout: int = 90) -> None:
    """Poll `predicate` until true or timeout — AOSS control-plane ops are async."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return
        except Exception:  # noqa: BLE001 — transient 404/409 while state flips
            pass
        time.sleep(3)
    print(f"  (timed out waiting for {what} — continuing)")


def chunk_words(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = CHUNK_WORDS - OVERLAP_WORDS
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if not piece:
            break
        out.append(" ".join(piece))
        if start + CHUNK_WORDS >= len(words):
            break
    return out


def iter_chunks():
    for corpus in CORPORA:
        for path in sorted((DATA_DIR / corpus).glob("*.md")):
            body = path.read_text(encoding="utf-8")
            key = f"{corpus}/{path.name}"
            for i, chunk in enumerate(chunk_words(body)):
                yield {
                    "text": chunk,
                    "source": path.name,
                    "key": key,
                    "corpus": corpus,
                    "audience": "all",
                    "chunk_id": f"{key}#{i}",
                }


def main() -> None:
    force = "--force" in sys.argv
    client = opensearch_client()

    # Chunking is cheap (no Bedrock) — do it first so we can compare counts.
    rows = list(iter_chunks())

    exists = client.indices.exists(index=INDEX_NAME)
    indexed = client.count(index=INDEX_NAME).get("count", 0) if exists else 0

    if not force and exists and indexed > 0:
        if indexed == len(rows):
            print(f"'{INDEX_NAME}' already has {indexed} chunks and data/ still produces "
                  f"{len(rows)} — nothing changed. Pass --force to rebuild anyway.")
            return
        print(f"Chunk count changed ({indexed} indexed -> {len(rows)} now) — rebuilding.")

    # AOSS delete/create/index are all eventually consistent. If we bulk before the
    # drop has fully propagated, the new docs can be swept away with the old ones
    # (index ends up empty); if the drop is skipped, bulk APPENDS under fresh
    # auto-ids and the count doubles. So: delete, wait until it's really gone,
    # recreate, wait until it's really there, then bulk — and verify the final count.
    if exists:
        client.indices.delete(index=INDEX_NAME)
        _wait(lambda: not client.indices.exists(index=INDEX_NAME), "drop to propagate")
        print(f"Dropped existing '{INDEX_NAME}'.")
    client.indices.create(index=INDEX_NAME, body=MAPPING)
    _wait(lambda: client.indices.exists(index=INDEX_NAME), "index to become visible")
    time.sleep(10)  # let the mapping settle before the first write
    print(f"Created '{INDEX_NAME}' with the knn_vector mapping.")

    print(f"Embedding {len(rows)} chunks (Titan)...")
    actions = []
    for i, row in enumerate(rows, 1):
        row["embedding"] = embed_text(row["text"])
        actions.append({"_index": INDEX_NAME, "_source": row})
        if i % 25 == 0 or i == len(rows):
            print(f"  embedded {i}/{len(rows)}")

    # Bulk, then confirm the docs actually stuck. A fresh AOSS index sometimes
    # accepts a bulk and then loses it while the mapping propagates — so if the
    # count doesn't reach the expected total, re-bulk (idempotent only because we
    # know the index is empty when count < total).
    target = len(rows)
    for attempt in range(4):
        try:
            helpers.bulk(client, actions, max_retries=3, initial_backoff=2, request_timeout=120)
        except Exception as e:  # noqa: BLE001 — 503s while the mapping propagates
            print(f"  bulk error, retrying ({e})")
            time.sleep(5 * (attempt + 1))
            continue
        persisted = 0
        for _ in range(18):
            time.sleep(5)
            persisted = client.count(index=INDEX_NAME).get("count", 0)
            if persisted >= target:
                break
        if persisted >= target:
            break
        print(f"  only {persisted}/{target} landed — re-bulking (attempt {attempt + 1})")
        if persisted:  # partial write: clear it so the re-bulk doesn't double
            client.indices.delete(index=INDEX_NAME)
            _wait(lambda: not client.indices.exists(index=INDEX_NAME), "drop to propagate")
            client.indices.create(index=INDEX_NAME, body=MAPPING)
            _wait(lambda: client.indices.exists(index=INDEX_NAME), "index to become visible")
            time.sleep(10)

    final = client.count(index=INDEX_NAME).get("count", 0)
    if final == target:
        print(f"Indexed {final} chunks into '{INDEX_NAME}'.")
    else:
        raise SystemExit(
            f"'{INDEX_NAME}' ended with {final} chunks, expected {target}. "
            "AOSS is still settling — recheck the count in a minute, or re-run with --force."
        )


if __name__ == "__main__":
    main()