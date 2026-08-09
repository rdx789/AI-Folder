"""Build one FAISS index per chunking strategy/param variant over the handbook,
each isolated at data/faiss_index_<strategy>/ so runs don't clobber each other.

    python chunking-study/build_indexes.py
"""
import json

import faiss
import numpy as np

from chunkers import chunk_fixed_size, chunk_semantic, chunk_separator, chunk_sentences
from common import DATA_DIR, EMBEDDING_MODEL_ID, CallTracker, embed_text, load_handbook_files

# Each entry: (strategy dir name, chunk-building function returning list[dict]).
# Fixed-size and sentence-based get several param variants per the assignment
# spec; separator and semantic have no tunable count, so one variant each.
STRATEGIES = [
    ("fixed_300w_50ov", lambda files, _t: chunk_fixed_size(files, 300, 50)),
    ("fixed_600w_100ov", lambda files, _t: chunk_fixed_size(files, 600, 100)),
    ("fixed_800w_200ov", lambda files, _t: chunk_fixed_size(files, 800, 200)),
    ("separator", lambda files, _t: chunk_separator(files)),
    ("sentence_3", lambda files, _t: chunk_sentences(files, 3)),
    ("sentence_5", lambda files, _t: chunk_sentences(files, 5)),
    ("sentence_8", lambda files, _t: chunk_sentences(files, 8)),
    ("semantic_llm", lambda files, t: chunk_semantic(files, t)),
]


def build_one(name: str, chunk_fn, files: list[tuple[str, str]]) -> dict:
    """Chunk, embed, and persist one strategy's FAISS index + metadata.json.
    Skips the build (returning {}) if the index already exists, mirroring
    ingest_kbs.py's idempotency so re-running the study doesn't re-spend on
    an index you already have — delete the dir to force a rebuild."""
    index_dir = DATA_DIR / f"faiss_index_{name}"
    if (index_dir / "index.faiss").exists():
        print(f"[{name}] index already exists at {index_dir.name}/ — skipping. "
              f"To rebuild, delete it first: rm -rf {index_dir}")
        return {}

    tracker = CallTracker(name)
    chunks = chunk_fn(files, tracker)
    if not chunks:
        raise SystemExit(f"[{name}] chunker produced 0 chunks — check the strategy implementation.")

    vectors = np.array([embed_text(c["text"], tracker) for c in chunks], dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # unit-normalized vectors -> cosine similarity
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "metadata.json").write_text(
        json.dumps({"embedding_model": EMBEDDING_MODEL_ID, "strategy": name, "chunks": chunks}, indent=2),
        encoding="utf-8",
    )

    print(f"[{name}] {len(chunks)} chunks -> {index_dir.name}/")
    tracker.print_summary()
    return tracker.summary()


def main() -> None:
    """Build every strategy in STRATEGIES over the handbook corpus and print a
    final embed/Nova-calls/$ comparison across the ones actually built this
    run (strategies skipped as already-built are left out of that table)."""
    files = load_handbook_files()
    print(f"Loaded {len(files)} handbook files.\n")

    summaries = []
    for name, chunk_fn in STRATEGIES:
        summary = build_one(name, chunk_fn, files)
        if summary:
            summaries.append(summary)
        print()

    if summaries:
        print("=== build summary ===")
        for s in summaries:
            print(
                f"{s['strategy']:<20} embed_calls={s['embed_calls']:<5} nova_calls={s['nova_calls']:<4} "
                f"cost=${s['cost_usd']:.6f}"
            )


if __name__ == "__main__":
    main()
