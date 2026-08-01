"""Bedrock Converse client — the only file that imports boto3."""

import os

import boto3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=get_region())
    return _client


def get_model_id() -> str:
    return os.environ["BEDROCK_MODEL_ID"]


def get_region() -> str:
    return os.environ["AWS_REGION"]


def call_model(messages, system, tool_config=None):
    kwargs = {
        "modelId": get_model_id(),
        "messages": messages,
        "system": system,
    }
    if tool_config:
        kwargs["toolConfig"] = tool_config
    return get_client().converse(**kwargs)