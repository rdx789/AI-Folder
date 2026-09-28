"""Provider and message helpers owned by the evaluation harness.

These small functions intentionally duplicate agent runtime behavior. Keeping them
here prevents eval modules from importing the student-facing agent implementation.
"""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv(find_dotenv())

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:9877/mcp")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_judge_model(**kwargs) -> ChatBedrockConverse:
    """Create the model used by LLM judges."""

    return ChatBedrockConverse(
        model=MODEL_ID,
        region_name=REGION,
        temperature=0,
        **kwargs,
    )


async def load_eval_tools() -> list:
    """Load the tool catalogue used to replay every agent stage."""

    client = MultiServerMCPClient(
        {"novaops": {"url": MCP_SERVER_URL, "transport": "streamable_http"}}
    )
    return await client.get_tools()


def message_text(message) -> str:
    """Flatten a message's string-or-content-block payload for scoring/reporting."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts).strip()
    return str(content)


def recent_turns(messages: list, keep: int = 2) -> list:
    """Return complete user-led turns when the harness extracts tool evidence."""

    boundaries = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    if not boundaries:
        return list(messages)
    start = boundaries[-keep] if len(boundaries) >= keep else boundaries[0]
    return list(messages[start:])
