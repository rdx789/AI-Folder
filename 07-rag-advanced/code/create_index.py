"""Creates the shared novaops-kb index (metadata + vector) inside the already-
provisioned OpenSearch Serverless collection.

By default this is a no-op if the index already exists (so re-running the pipeline
during testing never wipes a populated index by accident). Pass --force to drop and
recreate — only needed when the mapping/schema itself changed.
"""
import sys

from client import EMBED_DIM, INDEX_NAME, opensearch_client

MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    # faiss (not nmslib): nmslib is deprecated and not reliably
                    # available on OpenSearch Serverless. Titan embeds with
                    # normalize=True, so innerproduct == cosine similarity here.
                    "engine": "faiss",
                    "space_type": "innerproduct",
                    "parameters": {"ef_construction": 512, "m": 16},
                },
            },
            "source": {"type": "keyword"},       # full citation string, e.g. "handbook/severance.md"
            "key": {"type": "keyword"},          # corpus/filename — identifies the SOURCE DOC a chunk came from.
            "audience": {"type": "keyword"},     # "all" | "manager"
            "subjects": {"type": "keyword"},     # 1-3 tags from subjects.SUBJECTS
            "last_updated": {"type": "date"},
            "chunk_id": {"type": "keyword"},
        }
    },
}


def main():
    force = "--force" in sys.argv
    client = opensearch_client()
    exists = client.indices.exists(index=INDEX_NAME)
    if exists and not force:
        print(f"Index '{INDEX_NAME}' already exists — leaving it as-is. Pass --force to drop and recreate.")
        return
    if exists and force:
        client.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")
    client.indices.create(index=INDEX_NAME, body=MAPPING)
    print(f"Created index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()
