"""Parses every doc under data/handbook/ and data/manager_playbook/, chunks each
body at CHUNK_WORDS/OVERLAP_WORDS, tags each chunk with subjects (Nova), embeds
(Titan V2), and bulk-indexes into the shared novaops-kb index.

Chunking is cheap; embedding + tagging is real Bedrock cost. So by default this
chunks first and only re-ingests when the chunk count differs from what's
already indexed (chunk config or source docs changed). Pass --force to drop and
rebuild the index unconditionally.
"""
import re
import sys
import time
from pathlib import Path

from env_check import validate_env

validate_env()

from client import INDEX_NAME, embed_text, opensearch_client  # noqa: E402
from create_index import MAPPING  # noqa: E402  (reuse the one mapping definition)
from subjects import tag_subjects  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_WORDS = 600
OVERLAP_WORDS = 100

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_doc(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(f"{path} is missing YAML frontmatter")
    front_raw, body = match.groups()
    front = {}
    for line in front_raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        front[key.strip()] = value.strip()
    return front, body.strip()


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


def iter_docs():
    for sub in ("handbook", "manager_playbook"):
        for path in sorted((DATA_DIR / sub).glob("*.md")):
            yield path


def build_chunks() -> list[dict]:
    """Parse + chunk every doc. Cheap — no Bedrock calls — so we can count the
    chunks before deciding whether an expensive re-embed is warranted."""
    rows = []
    for path in iter_docs():
        front, body = parse_doc(path)
        key = f"{front['corpus']}/{path.name}"
        audience = front.get("audience", "all")
        last_updated = front.get("last_updated")
        chunks = chunk_words(body, CHUNK_WORDS, OVERLAP_WORDS)
        for i, chunk in enumerate(chunks):
            rows.append({
                "text": chunk, "key": key, "audience": audience,
                "last_updated": last_updated, "chunk_id": f"{key}#{i}",
            })
        print(f"{key}: {len(chunks)} chunks")
    return rows


def main():
    force = "--force" in sys.argv
    client = opensearch_client()
    exists = client.indices.exists(index=INDEX_NAME)
    count = client.count(index=INDEX_NAME).get("count", 0) if exists else 0

    rows = build_chunks()
    total = len(rows)

    # Re-embedding every chunk is real Bedrock spend. Only do it when something
    # actually changed: --force, an empty/missing index, or a different chunk
    # count (the cheap proxy for "chunk config or source docs changed").
    if exists and count > 0 and not force:
        if count == total:
            print(f"Index '{INDEX_NAME}' already has {count} docs and chunking still "
                  f"produces {total} — nothing to re-ingest. Pass --force to rebuild anyway.")
            return
        print(f"Chunk count changed ({count} indexed -> {total} now) — re-ingesting.")

    try:
        if exists:
            # AOSS Serverless has no _delete_by_query and auto-assigns _ids, so
            # dropping and recreating the index is the only way to clear it
            # without ending up with a second copy of every chunk.
            client.indices.delete(index=INDEX_NAME)
            client.indices.create(index=INDEX_NAME, body=MAPPING)
            print(f"Dropped and recreated '{INDEX_NAME}' ({count} old docs cleared)")
        else:
            # No index yet — create it with the real mapping so the bulk write
            # below doesn't trigger AOSS dynamic mapping (which would make
            # `embedding` a plain float array, not knn_vector).
            client.indices.create(index=INDEX_NAME, body=MAPPING)
            print(f"Created index '{INDEX_NAME}'")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not prepare '{INDEX_NAME}' for ingest ({e}).")

    # Now the expensive pass: one tag + one embed call per chunk.
    actions = []
    for row in rows:
        actions.append({
            "_index": INDEX_NAME,
            "_source": {
                "text": row["text"],
                "embedding": embed_text(row["text"]),
                "source": row["key"],
                "key": row["key"],
                "audience": row["audience"],
                "subjects": tag_subjects(row["text"]),
                "last_updated": row["last_updated"],
                "chunk_id": row["chunk_id"],
            },
        })

    # Bulk index. helpers.bulk raises BulkIndexError listing any docs the server
    # rejected — surface that instead of a silent short count. A fresh AOSS index
    # can 503 the first bulk while the mapping propagates, so retry the batch.
    from opensearchpy.helpers import bulk, BulkIndexError  # noqa: E402
    for attempt in range(4):
        try:
            indexed, _ = bulk(client, actions, max_retries=3, initial_backoff=2, request_timeout=120)
            break
        except BulkIndexError as e:
            raise SystemExit(f"{len(e.errors)} chunks rejected by OpenSearch: {e.errors[:2]}")
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  bulk retry {attempt + 1} ({e})")
            time.sleep(5 * (attempt + 1))

    # AOSS indexes asynchronously (no _refresh API); the count lags the write by
    # a few seconds. Poll so the "searchable" number is meaningful.
    persisted = 0
    for _ in range(12):
        time.sleep(5)
        persisted = client.count(index=INDEX_NAME).get("count", 0)
        if persisted >= indexed:
            break
    print(f"Bulk-indexed {indexed}/{total} chunks into '{INDEX_NAME}' ({persisted} searchable)")
    if persisted < indexed:
        print("NOTE: count still catching up — AOSS is eventually consistent; recheck in a minute.")


if __name__ == "__main__":
    main()
