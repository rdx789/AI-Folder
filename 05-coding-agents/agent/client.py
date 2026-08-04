"""Bedrock client seam — the only file that imports boto3."""
import os

import boto3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
    return _client


def get_model_id():
    return os.environ["BEDROCK_MODEL_ID"]
