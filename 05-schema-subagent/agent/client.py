"""Bedrock Converse client — the only place that imports boto3.

Keeping the client and model id behind these two functions is the provider
swap point: replacing Bedrock later means changing this file only.
"""
import os

import boto3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

_client = None


def _require_env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(f"missing required environment variable '{name}' — check your .env file") from None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=_require_env("AWS_REGION"),
        )
    return _client


def get_model_id() -> str:
    return _require_env("BEDROCK_MODEL_ID")
