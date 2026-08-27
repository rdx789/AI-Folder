"""Validates the .env vars this build needs before any script touches Bedrock or
OpenSearch. Per CLAUDE.md: never guess, default, or hard-code a missing value —
stop and ask.
"""
import os
import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

REQUIRED = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "BEDROCK_MODEL_ID",
    "BEDROCK_EMBEDDING_MODEL_ID",
    "OPENSEARCH_COLLECTION",
]


def validate_env() -> None:
    missing = [var for var in REQUIRED if not os.environ.get(var)]
    if missing:
        sys.exit(
            "Missing required .env vars: " + ", ".join(missing) +
            "\nAdd them to homework/.env before running this script."
        )
