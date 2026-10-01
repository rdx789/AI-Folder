"""End-to-end AWS retrieval and Bedrock replay through the read-only MCP server."""
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/evals/live.py): relative imports need the package, so
    # re-run this file as the installed module `maya.evals.live`.
    import runpy
    try:
        runpy.run_module('maya.evals.live', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit
import argparse
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from threading import Lock
import time

from ..backend import live_backend
from ..graph import Dependencies
from ..paths import DATA_PATH, RESULTS_DIR
from ..ingestion import index_body, indexed_corpus_matches, ingest, load_chunks
from ..model import BedrockMayaModel
from ..operations import MCPReadPort
from ..policy import READ_TOOLS
from ..ports import FakeWebexPort
from ..reranker import BedrockReranker
from ..retrieval import OpenSearchEvidenceRetriever
from .capture import build_recorded_graph
from .metering import MeteredBedrock, print_summary, summarize, write_requests
from .runner import load_session, run_evaluation



class BackendMeter:
    def __init__(self, client, embed, rerank):
        self.client, self.embedding, self.ranking = client, embed, rerank
        self.values = dict(embeddings=0, searches=0, reranks=0)
        self.lock = Lock()
        self.requests = []

    def counts(self):
        with self.lock:
            return dict(self.values)

    def bump(self, name):
        with self.lock:
            self.values[name] += 1

    def embed(self, query):
        self.bump('embeddings')
        return self.embedding(query)

    def search(self, **kwargs):
        self.bump('searches')
        result = self.client.search(**kwargs)
        self.requests.append({'index': kwargs['index'], 'body': kwargs['body'],
                              # Retriever validation owns malformed hits; metering must not mask it.
                              'returned_chunk_ids': [(h.get('_source') or {}).get('chunk_id')
                                 for h in result.get('hits', {}).get('hits', [])]})
        return result

    def rerank(self, query, candidates):
        self.bump('reranks')
        return self.ranking(query, candidates)


def prepare_index(client, embed, *, index, dimension, service, model, prepare):
    records = load_chunks(DATA_PATH)
    exists = client.indices.exists(index=index)
    print(f'OpenSearch index {index}: {"exists" if exists else "missing"}', flush=True)
    if not exists:
        if not prepare:
            raise ValueError('Missing Maya index; run with --prepare to create and ingest it')
        client.indices.create(index=index, body=index_body(
            dimension, serverless=service == 'aoss', embedding_model_id=model))
    # Serverless may acknowledge creation before its search endpoints see the index.
    from opensearchpy import NotFoundError
    creation_deadline = time.monotonic() + 90
    while True:
        try:
            client.count(index=index)
            break
        except NotFoundError:
            if time.monotonic() >= creation_deadline:
                raise
            time.sleep(3)
    mapping = client.indices.get_mapping(index=index)[index]['mappings']
    vector_field = mapping.get('properties', {}).get('vector')
    if not isinstance(vector_field, dict):
        raise ValueError(f'Index {index} has no vector field; it was not created by Maya')
    engine = vector_field.get('method', {}).get('engine')
    if engine == 'nmslib':
        raise ValueError('Index uses NMSLIB, which cannot enforce inline ACL filters; create a separate Faiss index')
    count = client.count(index=index)['count']
    added = 0
    if not count:
        if not prepare:
            raise ValueError('Empty Maya index; run with --prepare to ingest it')
        print(f'Embedding/indexing {len(records)} reviewed chunks.', flush=True)
        added = ingest(client, index=index, records=records, embed=embed, dimension=dimension,
                       serverless=service == 'aoss', embedding_model_id=model)
    elif not indexed_corpus_matches(client, index=index, records=records, dimension=dimension,
                                   count=count, embedding_model_id=model):
        raise ValueError('Existing index differs from the reviewed corpus; no existing data was replaced')
    deadline = time.monotonic() + 180
    while True:
        count = client.count(index=index)['count']
        if indexed_corpus_matches(client, index=index, records=records, dimension=dimension,
                                 count=count, embedding_model_id=model):
            print(f'OpenSearch ready: {count} searchable chunks, metadata and vectors verified.', flush=True)
            return {'index': index, 'expected_chunks': len(records), 'visible_chunks': count,
                    'corpus_matches': True, 'newly_indexed': added, 'dimension': dimension,
                    'embedding_model_id': model, 'vector_engine': engine or 'service-managed'}
        if time.monotonic() >= deadline:
            raise TimeoutError('Reviewed corpus not fully searchable after indexing')
        print(f'Waiting for index visibility: {count}/{len(records)} chunks.', flush=True)
        time.sleep(3)


@dataclass
class LiveRuntime:
    app: object
    capture: object
    webex: FakeWebexPort
    search: BackendMeter


async def main(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    client, embed, bedrock, dimension, service = await asyncio.to_thread(live_backend, MeteredBedrock)
    index = os.environ.get('MAYA_OPENSEARCH_INDEX', 'maya-evidence-v1')
    model_id = os.environ.get('BEDROCK_MODEL_ID')
    if not model_id:
        raise ValueError('Set BEDROCK_MODEL_ID in Maya .env')
    embedding_model = os.environ.get('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')
    readiness = await asyncio.to_thread(prepare_index, client, embed, index=index,
        dimension=dimension, service=service, model=embedding_model, prepare=args.prepare)
    meter = BackendMeter(client, embed, BedrockReranker(bedrock, model_id=model_id))
    retriever = OpenSearchEvidenceRetriever(meter, index=index, embed=meter.embed, rerank=meter.rerank)
    if args.probe:
        from ..schemas import CallerContext, ContextPlan
        evidence = await retriever.retrieve(caller=CallerContext('E004','UG_HR'), plan=ContextPlan(
            'policy_question', requires_retrieval=True,
            required_evidence=['saas_renewal_freeze_q3.md'], retrieval_query='Q3 Finance approval threshold'))
        print('Live retrieval probe:', [e.citation.source for e in evidence], flush=True)
        return 0
    params = StdioServerParameters(command=sys.executable, args=['-B', '-c',
        'from maya.server.server import mcp; mcp.run(transport="stdio")'])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            if names != READ_TOOLS:
                raise ValueError('MCP catalogue is not exactly the Maya read-only allowlist')
            print(f'MCP connected: {len(names)} read-only tools.', flush=True)
            port = FakeWebexPort()
            app, capture = build_recorded_graph(Dependencies(
                BedrockMayaModel(bedrock, model_id=model_id), retriever, port, MCPReadPort(session)))
            bedrock.sink = capture.llm  # per-turn model/embedding usage
            report = await run_evaluation(LiveRuntime(app, capture, port, meter), load_session(),
                                          thread_id=args.thread_id)
    report.update(mode='live', model_id=model_id, index_readiness=readiness,
                  quality_scope='Live Bedrock planning/answers/Titan embeddings/reranking; live AWS OpenSearch; local read-only MCP transport; fake Webex.',
                  mcp_tools=sorted(names), backend_counts=meter.counts(),
                  opensearch_requests=meter.requests,
                  token_summary=summarize([*report['records'], report['replay'], report['first_request']]))
    print_summary(report['token_summary'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_requests(bedrock.requests, args.output)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(f'Report: {args.output}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Create/ingest a missing or empty Maya index; never replace existing data')
    parser.add_argument('--probe', action='store_true', help='Check one real authorized retrieval before running the session')
    parser.add_argument('--thread-id', default=None)
    parser.add_argument('--output', type=Path, default=RESULTS_DIR / 'latest-s2-live.json')
    from ..backend import external_errors
    try:
        raise SystemExit(asyncio.run(main(parser.parse_args())))
    except external_errors() as exc:
        raise SystemExit(f'Live replay failed: {type(exc).__name__}: {exc}')
