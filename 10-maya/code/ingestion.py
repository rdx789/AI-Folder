"""Allowlisted Markdown ingestion with reviewed, fail-closed access metadata."""
import hashlib
import json
import re
import math
from datetime import date
from pathlib import Path

from .access import COLLECTIONS, GROUPS

DEFAULT_MANIFEST = Path(__file__).with_name('access_manifest.json')


def index_body(dimension: int, *, serverless: bool = False, embedding_model_id: str | None = None,
               engine: str = 'faiss') -> dict:
    if dimension < 1:
        raise ValueError('Embedding dimension must be positive')
    method = {'name': 'hnsw', 'space_type': 'innerproduct'}
    # Classic Serverless defaults to NMSLIB, which cannot execute inline ACL filters.
    # NextGen deployments that choose their own engine may explicitly use 'auto'.
    if engine not in {'faiss', 'auto'} or (engine == 'auto' and not serverless):
        raise ValueError('Use faiss, or auto for NextGen Serverless')
    if engine != 'auto':
        method['engine'] = engine
    properties = {k: {'type': 'keyword'} for k in (
        'source', 'source_id', 'chunk_id', 'collection', 'audience',
        'subject_employee_id', 'sensitivity', 'allowed_employee_ids')}
    properties.update(text={'type': 'text'}, self_service={'type': 'boolean'},
                      updated_at={'type': 'date', 'format': 'strict_date'},
                      vector={'type': 'knn_vector', 'dimension': dimension, 'method': method})
    mappings = {'dynamic': 'strict', 'properties': properties}
    if embedding_model_id:
        mappings['_meta'] = {'embedding_model_id': embedding_model_id}
    return {'settings': {'index': {'knn': True}}, 'mappings': mappings}


def chunk_document(text: str, *, words: int = 250, overlap: int = 50) -> list[str]:
    if not 0 <= overlap < words:
        raise ValueError('Require 0 <= overlap < words')
    tokens = text.split()
    chunks = []
    for start in range(0, len(tokens), words - overlap):
        chunks.append(' '.join(tokens[start:start + words]))
        if start + words >= len(tokens):
            break
    return chunks


def validate_metadata(metadata: dict) -> None:
    audience = metadata.get('audience')
    allowed = metadata.get('allowed_employee_ids')
    subject = metadata.get('subject_employee_id')
    if (not isinstance(audience, list) or not audience or not set(audience) <= GROUPS
            or metadata.get('sensitivity') not in {'internal', 'confidential', 'restricted'}
            or type(metadata.get('self_service')) is not bool
            or not isinstance(allowed, list)
            or any(not isinstance(x, str) or not re.fullmatch(r'E\d{3}', x) for x in allowed)
            or (subject is not None and (not isinstance(subject, str) or not re.fullmatch(r'E\d{3}', subject)))):
        raise ValueError('Invalid access metadata')
    if metadata['self_service'] and (subject is None or allowed != [subject]):
        raise ValueError('Self-service requires exactly the named subject')
    if subject is not None and 'UG_REGULAR' in audience:
        raise ValueError('Person-specific documents cannot be group-visible to regular employees')
    if metadata.get('updated_at') is not None:
        date.fromisoformat(metadata['updated_at'])


def load_chunks(data_root: Path, *, manifest_path: Path = DEFAULT_MANIFEST) -> list[dict]:
    root = Path(data_root).resolve(strict=True)
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read access manifest {manifest_path}: {exc}') from exc
    if not isinstance(manifest, dict):
        raise ValueError(f'Access manifest {manifest_path} must be a JSON object')
    records = []
    for collection in COLLECTIONS:
        folder = root / collection
        if not folder.exists():
            continue
        for path in sorted(folder.rglob('*.md')):
            if not path.resolve().is_relative_to(root):
                raise ValueError('Dataset symlink escapes configured root')
            source = path.relative_to(root).as_posix()
            if source not in manifest:
                raise ValueError(f'Missing reviewed access metadata: {source}')
            metadata = manifest[source]
            # Still fail closed, but say which reviewed document is at fault.
            try:
                if not isinstance(metadata, dict):
                    raise ValueError('Invalid access metadata')
                validate_metadata(metadata)
                text = path.read_text(encoding='utf-8')
                updated = metadata.get('updated_at')
                if not updated:
                    match = re.search(r'^(?:Last updated|Date|Effective date):\s*(\d{4}-\d{2}-\d{2})\s*$', text, re.M)
                    updated = match.group(1) if match else None
                if updated:
                    date.fromisoformat(updated)
            except (ValueError, OSError) as exc:
                raise ValueError(f'{source}: {exc}') from exc
            # Source ID survives moves of the configured root; chunk ID is content-addressed.
            source_id = hashlib.sha256(source.encode()).hexdigest()
            for number, chunk in enumerate(chunk_document(text)):
                chunk_id = hashlib.sha256(f'{source_id}:{number}:{chunk}'.encode()).hexdigest()
                record = dict(text=chunk, source=source, source_id=source_id,
                              chunk_id=chunk_id, collection=collection, **metadata)
                if updated:
                    record['updated_at'] = updated
                else:
                    record.pop('updated_at', None)
                records.append(record)
    if not records:
        raise ValueError('No allowlisted documents found')
    return records


def indexed_corpus_matches(client, *, index: str, records: list[dict], dimension: int,
                           count: int, embedding_model_id: str | None = None) -> bool:
    if count != len(records) or count > 10000:
        return False
    if embedding_model_id:
        mappings = client.indices.get_mapping(index=index)[index]['mappings']
        if mappings.get('_meta', {}).get('embedding_model_id') != embedding_model_id:
            return False
    response = client.search(index=index, body={
        'size': count, 'query': {'match_all': {}},
        '_source': sorted({key for record in records for key in record} | {'vector'}),
    })
    if response.get('timed_out') or response.get('_shards', {}).get('failed', 0):
        return False
    indexed = [hit['_source'] for hit in response['hits']['hits']]
    if len(indexed) != count:
        return False
    expected = {r['chunk_id']: r for r in records}
    seen = set()
    for stored in indexed:
        stored = dict(stored)
        vector = stored.pop('vector', None)
        chunk_id = stored.get('chunk_id')
        if (chunk_id in seen or stored != expected.get(chunk_id)
                or not isinstance(vector, list) or len(vector) != dimension
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
            return False
        seen.add(chunk_id)
    return True


def ingest(client, *, index: str, records: list[dict], embed, dimension: int,
           force: bool = False, serverless: bool = False,
           embedding_model_id: str | None = None) -> int:
    """Skip unchanged indexed evidence; force regenerates vectors and rebuilds index.

    All embeddings complete before replacing the index. Serverless assigns storage
    IDs; stable citation IDs remain in metadata. This never deletes a collection.
    """
    if not records:
        raise ValueError('No evidence records supplied')
    existing_count = client.count(index=index)['count']
    if existing_count and not force:
        if indexed_corpus_matches(client, index=index, records=records, dimension=dimension,
                                  count=existing_count, embedding_model_id=embedding_model_id):
            return 0
        raise ValueError('Indexed evidence or embedding model differs; rerun with --force')
    operations = []
    for record in records:
        vector = embed(record['text'])
        if (len(vector) != dimension
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
            raise ValueError('Invalid embedding or dimension mismatch')
        operations.extend([{'index': {'_index': index}}, dict(record, vector=vector)])
    if force:
        client.indices.delete(index=index)
        client.indices.create(index=index, body=index_body(
            dimension, serverless=serverless, embedding_model_id=embedding_model_id))
    response = client.bulk(body=operations)
    if response.get('errors', False):
        raise RuntimeError('Evidence ingestion failed; rerun with --force to rebuild')
    return len(records)
