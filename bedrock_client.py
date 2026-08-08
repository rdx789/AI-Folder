"""Single point of contact with Bedrock — swap providers by editing only this file."""
import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

_REGION = os.environ["AWS_REGION"]
_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
_EMBEDDING_MODEL_ID = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]
_EMBEDDING_DIMENSIONS = int(os.getenv("BEDROCK_EMBEDDING_DIMENSIONS", "1024"))

_client = boto3.client("bedrock-runtime", region_name=_REGION)


def converse(messages, system=None, tool_config=None, tool_choice=None, max_tokens=1024, temperature=0.0):
    """One function wraps every chat-style Bedrock call (text or tool-use)."""
    kwargs = {
        "modelId": _MODEL_ID,
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    if tool_config:
        cfg = {"tools": tool_config}
        if tool_choice:
            cfg["toolChoice"] = tool_choice
        kwargs["toolConfig"] = cfg
    try:
        return _client.converse(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Bedrock converse call failed: {exc}") from exc


def embed(text, dimensions=None):
    """Embed a single string with the Titan embedding model. Returns a list[float]."""
    try:
        resp = _client.invoke_model(
            modelId=_EMBEDDING_MODEL_ID,
            body=json.dumps({
                "inputText": text,
                "dimensions": dimensions or _EMBEDDING_DIMENSIONS,
                "normalize": True,
            }),
        )
        return json.loads(resp["body"].read())["embedding"]
    except Exception as exc:
        raise RuntimeError(f"Bedrock embedding call failed: {exc}") from exc


DATA_DIR = Path(__file__).resolve().parent / "data"
EMBEDDING_DIMENSIONS = _EMBEDDING_DIMENSIONS
EMBEDDING_MODEL_ID = _EMBEDDING_MODEL_ID