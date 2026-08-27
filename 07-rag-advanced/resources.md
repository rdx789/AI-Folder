# Resources — Lesson 7 (RAG Advanced)

Further reading, grouped by part. All optional.

## OpenSearch + vector search (Part 1)

- [OpenSearch k-NN / vector search](https://opensearch.org/docs/latest/search-plugins/knn/index/) — index types, the `knn_vector` field, and engines (faiss/nmslib/lucene).
- [Amazon OpenSearch Serverless — vector collections](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-vector-search.html) — the managed store this lab uses.
- [HNSW paper (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320) — the approximate-nearest-neighbour graph, if you want the algorithm behind "approximate."
- [OpenSearch Serverless pricing](https://aws.amazon.com/opensearch-service/pricing/) — the OCU-hour model, and how NextGen's scale-to-zero changes it (see Part 1).

## Filtering (Part 2)

- [Filtered k-NN in OpenSearch](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/) — how the `filter` inside a k-NN query works (and why the engine matters).
- [Bedrock Converse tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html) — the forced-tool pattern behind the subject tagger and planner.

## Reranking (Part 3)

- [Cross-encoders vs. bi-encoders (Sentence-Transformers)](https://www.sbert.net/examples/applications/cross-encoder/README.html) — why a reranker is a stronger (and pricier) relevance signal than the embedding used for retrieval.
- [Amazon Bedrock Rerank API](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank.html) — a managed reranker (Cohere/Amazon), if you want to swap the Nova reranker for a dedicated one.

## RAG evaluation (Parts 2 & 3)

- [RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/) — faithfulness and context relevance as standard, framework-blessed RAG metrics (this lab implements them by hand with an LLM judge).
- [Retrieval metrics: Recall@k, Precision@k, MRR](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) — the labeled metrics the homework adds on top of the LLM judges.
