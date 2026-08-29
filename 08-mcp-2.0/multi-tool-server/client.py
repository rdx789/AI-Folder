"""
Bedrock wiring for the MCP agent.

Everything provider-specific lives here so agent.py can stay about the MCP loop,
not about Bedrock. The one job that matters is mcp_tools_to_bedrock(): it bridges
the two tool formats — MCP advertises tools one way, the Bedrock Converse API
wants them another. MCP itself is provider-neutral; this function is the seam
you'd rewrite to drive the same MCP server from OpenAI or Anthropic instead.

Adding tools to the server never touches this file — that's the whole point.
"""

import os

import boto3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")


def get_client():
    """boto3 Bedrock runtime client (SigV4 auth from the AWS_* env vars)."""
    return boto3.client("bedrock-runtime", region_name=REGION)


def mcp_tools_to_bedrock(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions into the Bedrock Converse `toolConfig` shape.

    MCP gives us name / description / input_schema (a JSON Schema dict; the field is
    snake_case in MCP 2.0, camelCase inputSchema on 1.x). Bedrock wants each tool
    wrapped as {"toolSpec": {..., "inputSchema": {"json": <schema>}}} — note that outer
    "inputSchema" is Bedrock's own key, not the MCP one. Same information, different
    envelope — this is the only provider-specific glue in the whole agent.
    """
    return [
        {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "inputSchema": {"json": tool.input_schema},
            }
        }
        for tool in mcp_tools
    ]