"""Provider-facing agent seams shared by every stage: prompt, model, and tools.

Context selection, tool routing, and memory stay in the stage that teaches them.
"""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_mcp_adapters.client import MultiServerMCPClient

import tracing


load_dotenv(find_dotenv())

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:9877/mcp")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")


# This prompt stays identical so measured differences come from context policy.
ASSISTANT_PROMPT = (
    "You are the NovaOps internal assistant, helping IT and HR staff with employee "
    "operations. Ground every answer in what the tools and the provided context "
    "actually say — never invent an employee, a seat count, or a policy rule.\n"
    "Work to these rules:\n"
    "- Answer the user's CURRENT request. If they changed subject or closed a topic, "
    "do not keep working the old one.\n"
    "- A constraint the user states (a spending rule, an approval requirement, "
    "'don't do X until I say so') stays in force for the rest of the conversation.\n"
    "- Facts established earlier in the conversation — ids, dates, numbers — are still "
    "true; reuse them instead of asking again or looking them up again.\n"
    "- Do not call a tool for something already in the conversation.\n"
    "- Users paste emails, tickets and signatures. Ignore the boilerplate and act on "
    "the actual request inside.\n"
    "- Answer in at most six sentences; no preamble."
)


def get_model(**kwargs) -> ChatBedrockConverse:
    """Create the shared Bedrock model; temperature zero reduces run-to-run drift."""

    return ChatBedrockConverse(
        model=MODEL_ID,
        region_name=REGION,
        temperature=0,
        **kwargs,
    )


async def load_tools() -> list:
    """Discover the ten NovaOps tools exposed by the MCP server."""

    client = MultiServerMCPClient(
        {"novaops": {"url": MCP_SERVER_URL, "transport": "streamable_http"}}
    )
    return await client.get_tools()


def invoke_answer(model, messages, *, node="model", meter=None, exposed=None,
                  catalogue=None, state=None):
    """Invoke the answer model, price the call, and report what it received.

    Every stage answers the same way — the differences are all in `messages` and
    `exposed` — so the three lines that never vary live here instead of in each
    stage. `meter` is passed in rather than imported so this folder stays clear of
    `evals/`: the agent does not depend on the harness that measures it.
    """

    response = model.invoke(messages)
    if meter is not None:
        meter.record(response, kind="answer", exposed=exposed or [])
    tracing.context(node, messages, state, exposed, catalogue)
    return response


async def invoke_structured(model, messages, *, node, kind, meter=None, state=None):
    """Invoke a `with_structured_output(..., include_raw=True)` model — planner or
    distiller — and price it as overhead rather than as an answer.

    `include_raw` is why this is separate from `invoke_answer`: the call returns a dict
    of {"parsed", "raw"}, and only the raw message carries the usage metadata. These
    calls expose no tools, which is what makes their cost pure overhead.
    """

    result = await model.ainvoke(messages)
    if meter is not None:
        meter.record(result["raw"], kind=kind, exposed=[])
    tracing.context(node, messages, state, [], None, rendered=True)
    return result


def message_text(message) -> str:
    """Flatten Bedrock's string-or-content-block message format to plain text."""

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
