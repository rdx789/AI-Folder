"""Optional live AWS wiring. Importing Maya never connects or loads credentials."""
import json
import os
from urllib.parse import urlparse

from .reranker import BedrockReranker
from .retrieval import OpenSearchEvidenceRetriever


def external_errors() -> tuple[type[BaseException], ...]:
    """Configuration, filesystem, AWS and OpenSearch failures a CLI reports cleanly."""
    errors = [ValueError, OSError, RuntimeError]
    try:
        from botocore.exceptions import BotoCoreError, ClientError
        errors += [BotoCoreError, ClientError]
    except ImportError:
        pass
    try:
        from opensearchpy.exceptions import OpenSearchException
        errors.append(OpenSearchException)
    except ImportError:
        pass
    return tuple(errors)


def live_backend(wrap_bedrock=None):
    from dotenv import load_dotenv
    from .paths import ENV_FILE
    load_dotenv(ENV_FILE, override=False)
    import boto3
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

    region = os.environ.get('AWS_REGION', 'us-east-1')
    service = os.environ.get('OPENSEARCH_SERVICE', 'aoss')
    if service not in ('aoss', 'es'):
        raise ValueError('OPENSEARCH_SERVICE must be aoss or es')
    session = boto3.Session(region_name=region)
    search_session = session
    if os.environ.get('OPENSEARCH_AWS_ACCESS_KEY_ID'):
        if not os.environ.get('OPENSEARCH_AWS_SECRET_ACCESS_KEY'):
            raise ValueError('OPENSEARCH_AWS_ACCESS_KEY_ID is set without OPENSEARCH_AWS_SECRET_ACCESS_KEY')
        search_session = boto3.Session(
            aws_access_key_id=os.environ['OPENSEARCH_AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['OPENSEARCH_AWS_SECRET_ACCESS_KEY'],
            aws_session_token=os.environ.get('OPENSEARCH_AWS_SESSION_TOKEN'),
            region_name=region,
        )
    endpoint = resolve_endpoint(search_session, service=service)
    credentials = search_session.get_credentials()
    if credentials is None:
        raise ValueError('No AWS credentials found for OpenSearch; check .env or your AWS profile')
    parsed = urlparse(endpoint)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.path not in ('', '/'):
        raise ValueError('OPENSEARCH_ENDPOINT must be an HTTPS origin')
    client = OpenSearch(
        hosts=[{'host': parsed.hostname, 'port': parsed.port or 443}],
        http_auth=AWSV4SignerAuth(credentials, region, service),
        use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection,
        timeout=120, max_retries=3, retry_on_timeout=True,
    )
    bedrock = session.client('bedrock-runtime')
    if wrap_bedrock is not None:
        bedrock = wrap_bedrock(bedrock)  # e.g. evals.metering.MeteredBedrock
    try:
        dimension = int(os.environ.get('MAYA_EMBED_DIM', '1024'))
    except ValueError:
        raise ValueError('MAYA_EMBED_DIM must be an integer') from None
    model = os.environ.get('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')

    def embed(text):
        response = bedrock.invoke_model(modelId=model, body=json.dumps({
            'inputText': text, 'dimensions': dimension, 'normalize': True,
        }))
        try:
            vector = json.loads(response['body'].read())['embedding']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Malformed embedding response') from exc
        if not isinstance(vector, list) or len(vector) != dimension:
            raise ValueError('Embedding dimension mismatch')
        return vector

    return client, embed, bedrock, dimension, service


def live_retriever() -> OpenSearchEvidenceRetriever:
    client, embed, bedrock, _, _ = live_backend()
    if not os.environ.get('BEDROCK_MODEL_ID'):
        raise ValueError('Set BEDROCK_MODEL_ID in Maya .env (used by the reranker)')
    return OpenSearchEvidenceRetriever(
        client, index=os.environ.get('MAYA_OPENSEARCH_INDEX', 'maya-evidence-v1'),
        embed=embed, rerank=BedrockReranker(bedrock, model_id=os.environ['BEDROCK_MODEL_ID']),
    )


def resolve_endpoint(session, *, service: str) -> str:
    """Lesson 7: explicit HTTPS endpoint or current Serverless collection endpoint."""
    override = os.environ.get('OPENSEARCH_ENDPOINT')
    if override:
        return override
    name = os.environ.get('OPENSEARCH_COLLECTION')
    if service != 'aoss' or not name:
        raise ValueError('Set OPENSEARCH_ENDPOINT or OPENSEARCH_COLLECTION in Maya .env')
    from botocore.exceptions import BotoCoreError, ClientError
    try:
        details = session.client('opensearchserverless').batch_get_collection(
            names=[name]).get('collectionDetails', [])
    except (BotoCoreError, ClientError) as exc:
        raise ValueError(f'Could not resolve OpenSearch collection {name!r}: {exc}') from exc
    if not details or details[0].get('status') != 'ACTIVE' or not details[0].get('collectionEndpoint'):
        raise ValueError('OpenSearch collection is missing or not ACTIVE')
    return details[0]['collectionEndpoint']
