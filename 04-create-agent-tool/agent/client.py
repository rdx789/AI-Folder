"""Provider seam — the only place that imports boto3.

Swap get_client() / get_model_id() here to change the LLM backend without
touching the loop or tools.
"""
import os

import boto3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def get_model_id() -> str:
    model = os.environ.get("BEDROCK_MODEL_ID")
    if not model:
        raise RuntimeError("BEDROCK_MODEL_ID is not set")
    return model


def get_client():
    region = os.environ.get("AWS_REGION")
    if not region:
        raise RuntimeError("AWS_REGION is not set")
    return boto3.client("bedrock-runtime", region_name=region)


def call_model(client, model_id: str, system: str, messages: list, tools: list) -> dict:
    """Single Bedrock Converse call. Returns the raw API response."""
    kwargs = {
        "modelId": model_id,
        "system": [{"text": system}],
        "messages": messages,
    }
    if tools:
        kwargs["toolConfig"] = {"tools": tools}
    return client.converse(**kwargs)
