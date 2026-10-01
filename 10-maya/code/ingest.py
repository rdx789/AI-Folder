"""python -m maya.ingest --data-root data [--create-index] [--ingest]."""
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/ingest.py): relative imports need the package, so
    # re-run this file as the installed module `maya.ingest`.
    import runpy
    try:
        runpy.run_module('maya.ingest', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit
import argparse
from collections import Counter
import os
from pathlib import Path

from .ingestion import DEFAULT_MANIFEST, index_body, ingest, load_chunks
from .paths import DATA_PATH, ENV_FILE


def main():
    from .backend import external_errors
    try:
        _main()
    except external_errors() as exc:
        raise SystemExit(f'maya-ingest failed: {type(exc).__name__}: {exc}')


def _main():
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE, override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_PATH)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--create-index', action='store_true')
    parser.add_argument('--ingest', action='store_true')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate all embeddings and rebuild the configured Maya index')
    args = parser.parse_args()
    args.ingest = args.ingest or args.force
    records = load_chunks(args.data_root, manifest_path=args.manifest)
    counts = Counter(c['collection'] for c in records)
    print(f'Validated {len(records)} chunks: {dict(counts)}')
    if not counts['contracts']:
        print('Contracts absent; Webex KB/memo evidence is available. Contract requirements remain unresolved.')
    if not (args.create_index or args.ingest):
        print('Dry run; no embedding or OpenSearch calls.')
        return
    from .backend import live_backend
    client, embed, _, dimension, service = live_backend()
    index = os.environ.get('MAYA_OPENSEARCH_INDEX', 'maya-evidence-v1')
    model = os.environ.get('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=index_body(
            dimension, serverless=service == 'aoss', embedding_model_id=model))
    if args.ingest:
        added = ingest(client, index=index, records=records, embed=embed, dimension=dimension,
                       force=args.force, serverless=service == 'aoss', embedding_model_id=model)
        if added == 0:
            print('Evidence and vectors are unchanged; skipping ingestion.')
            return
        print(f'Indexed {added} chunks.')
        print('Serverless indexing is eventually visible; verify readiness before using retrieval.')


if __name__ == '__main__':
    main()
