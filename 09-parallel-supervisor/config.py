"""Configuration: load the environment once, expose the LLM factory and MCP URL."""

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

_REQUIRED = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "BEDROCK_MODEL_ID",
    "BEDROCK_EMBEDDING_MODEL_ID",
)

_missing = [name for name in _REQUIRED if not os.environ.get(name)]
if _missing:
    raise RuntimeError(
        "Missing required environment variable(s): "
        + ", ".join(_missing)
        + ". Set them in your .env before running."
    )

# Where the provided NovaOps MCP server listens. Overridable for a non-local server.
MCP_SERVER_URL = os.environ.get("NOVAOPS_MCP_URL", "http://127.0.0.1:9876/mcp")


def get_llm(temperature: float = 0.0):
    """Single construction point for the chat model, so the provider stays swappable."""
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.environ["AWS_REGION"],
        temperature=temperature,
    )
