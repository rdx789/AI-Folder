"""Ingest every corpus under a root folder — one FAISS index per subfolder.

A step up from 01-simple-rag/ingest.py: instead of one hard-coded corpus, this
DISCOVERS the knowledge bases under data/ — each subfolder of .md files is one
KB — and builds a separate index for each, at data/faiss_index_<name>/.

    python 02-rag-as-tool/ingest_kbs.py             # every KB under ./data
    python 02-rag-as-tool/ingest_kbs.py some/root   # or point it at another root

rag_by_role.py then serves those indexes as per-audience search tools. This is the
naive answer to multi-audience RAG: one index per corpus. It works — but the
`for kb in kbs` loop in main() is exactly the cost. Every audience is a wholly
separate store to build, hold, and re-embed on every edit, and no single search
can rank across two corpora — the model has to pick an index first. Lesson 7
replaces the N indexes with ONE index plus a metadata filter.
"""
import json
import os
import sys
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_DIR / "data"

CHUNK_WORDS = 250
OVERLAP_WORDS = 50

client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
EMBEDDING_MODEL_ID = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]


# --- the pipeline stages — same as 01-simple-rag/ingest.py, but parameterized
#     by folder so one set of functions serves every KB -------------------------

def load_documents(docs_dir: Path) -> list[str]:
    return [path.read_text(encoding="utf-8") for path in sorted(docs_dir.glob("*.md"))]


def chunk_document(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def chunk_documents(documents: list[str]) -> list[str]:
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    print(f"  chunked {len(documents)} documents -> {len(chunks)} chunks")
    return chunks


def embed_text(text: str) -> list[float]:
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def embed_chunks(chunks: list[str]) -> np.ndarray:
    embeddings = []
    for i, chunk in enumerate(chunks, start=1):
        embeddings.append(embed_text(chunk))
        if i % 20 == 0 or i == len(chunks):
            print(f"  embedded {i}/{len(chunks)}")
    return np.array(embeddings, dtype="float32")


def save_index(vectors: np.ndarray, chunks: list[str], index_dir: Path) -> None:
    # IndexFlatIP + unit-normalized vectors = cosine-similarity search (see
    # 01-simple-rag/ingest.py for the why).
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "metadata.json").write_text(
        json.dumps({"embedding_model": EMBEDDING_MODEL_ID, "chunks": chunks}, indent=2),
        encoding="utf-8",
    )
    print(f"  saved {index.ntotal} vectors (dim {vectors.shape[1]}) -> {index_dir.name}/")


# --- discover the KBs, then build one index per KB -----------------------------

def map_kbs(root: Path) -> list[dict]:
    """Find each knowledge base under root: one subfolder of .md files = one KB.

    Only one level down is scanned, and a subfolder counts only if it directly
    holds .md files — so the generated faiss_index_* dirs (which hold no .md)
    are skipped automatically, and re-running is safe.
    """
    kbs = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if not any(sub.glob("*.md")):
            continue
        kbs.append({"name": sub.name, "docs_dir": sub, "index_dir": root / f"faiss_index_{sub.name}"})
    return kbs


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DATA_DIR
    kbs = map_kbs(root)
    if not kbs:
        raise SystemExit(f"No knowledge bases under {root} — expected subfolders containing .md files.")

    print(f"Found {len(kbs)} knowledge base(s): {', '.join(kb['name'] for kb in kbs)}\n")
    for kb in kbs:
        print(f"[{kb['name']}]")
        # Embedding is the slow, paid step, so we skip a KB whose index already
        # exists (e.g. faiss_index_handbook, already built in 01-simple-rag). To
        # force a rebuild after editing a corpus, delete its index_dir and re-run.
        if (kb["index_dir"] / "index.faiss").exists():
            print(f"  index already exists at {kb['index_dir'].name}/ — skipping.")
            print(f"  to rebuild, delete it first:  rm -rf {kb['index_dir']}\n")
            continue
        documents = load_documents(kb["docs_dir"])
        chunks = chunk_documents(documents)
        vectors = embed_chunks(chunks)
        save_index(vectors, chunks, kb["index_dir"])
        print()


if __name__ == "__main__":
    main()
