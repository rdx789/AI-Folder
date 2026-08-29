"""
A Bedrock agent that answers questions using MCP tools (Exercise 2).

The loop is the same tool-use loop you saw in the agents lesson — but the tools
are not local functions. They live on the MCP server and are discovered at connect
time. The model decides which discovered tool to call; we run the call over MCP and
feed the result back, until the model gives a final answer.

Run (with server.py running in another terminal):
  python agent.py "How can I work from home securely?"   # one-shot
  python agent.py                                          # interactive REPL
"""

import asyncio
import os
import sys

from mcp import Client

from client import MODEL_ID, get_client, mcp_tools_to_bedrock

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:9876/mcp")
MAX_TURNS = 10

# The agent's instructions — kept right here, not in client.py, so this file is the
# whole agent in one read. (eval.py keeps its own copy so it scores this same prompt.)
SYSTEM_PROMPT = (
    "You are the NovaOps internal assistant. Answer employee questions using the "
    "MCP tools available to you. Discover what you need — list things before you "
    "fetch them — then give a clear, final answer grounded in what the tools return."
)


def result_text(result) -> str:
    """Join the text of EVERY content block in an MCP tool result.

    A tool that returns a list (e.g. list_policies) comes back as one text block
    PER item, so reading only content[0] would hand the model just the first item
    and silently drop the rest. Join them all."""
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def final_text(message: dict) -> str:
    """Pull the model's plain-text answer out of a Converse message."""
    return "".join(b["text"] for b in message["content"] if "text" in b).strip()


async def run_tool_calls(client: Client, message: dict) -> list[dict]:
    """Execute every tool the model asked for in this message, over MCP.

    Returns one toolResult block per toolUse, to send back as the next user turn.
    This is the half of the loop that actually touches the server — answer() below
    is just the turn-taking skeleton around it.
    """
    tool_results = []
    for block in message["content"]:
        if "toolUse" not in block:
            continue
        call = block["toolUse"]
        print(f"  → MCP tool: {call['name']}({call['input']})")
        result = await client.call_tool(call["name"], call["input"])
        tool_results.append(
            {
                "toolResult": {
                    "toolUseId": call["toolUseId"],
                    "content": [{"text": result_text(result)}],
                    # Tell the model when a tool errored so it can recover.
                    **({"status": "error"} if result.is_error else {}),
                }
            }
        )
    return tool_results


async def answer(client: Client, tool_config: dict, question: str) -> str:
    """Run the tool-use loop for one question until the model gives a final answer.

    The loop is only three moves: ask the model; if it wants tools, run them and
    feed the results back; otherwise return its answer.
    """
    bedrock = get_client()
    messages = [{"role": "user", "content": [{"text": question}]}]

    for _ in range(MAX_TURNS):
        # boto3 is synchronous; calling it inside this async function just blocks
        # briefly. That's fine for a single-user REPL — no executor needed. (The
        # async matters for the MCP calls in run_tool_calls, which ARE awaited.)
        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=tool_config,
        )
        output = response["output"]["message"]
        messages.append(output)  # keep the model's turn (incl. its tool requests)

        if response["stopReason"] != "tool_use":
            return final_text(output)  # end_turn: the model is done — return its text.

        # The model requested tools: run them and send the results back as a user
        # turn. That's what closes the loop.
        messages.append({"role": "user", "content": await run_tool_calls(client, output)})

    return "(Reached the tool-call limit without a final answer.)"


async def main() -> None:
    # The high-level Client connects and runs the initialize handshake on context entry
    # (MCP 2.0). One open client serves the whole session — every tool call the agent
    # makes below goes through it.
    async with Client(MCP_SERVER_URL) as client:
        # Discover the server's tools and translate them into Bedrock's format.
        tools = (await client.list_tools()).tools
        tool_config = {"tools": mcp_tools_to_bedrock(tools)}
        print(f"Connected. Tools: {', '.join(t.name for t in tools)}\n")

        question = " ".join(sys.argv[1:]).strip()
        if question:  # one-shot mode (handy for scripting / testing)
            print(f"You: {question}")
            print(f"\nAssistant:\n{await answer(client, tool_config, question)}")
            return

        while True:  # REPL mode
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if q.lower() in {"exit", "quit", "q"}:
                break
            if q:
                print(f"\nAssistant:\n{await answer(client, tool_config, q)}\n")


if __name__ == "__main__":
    asyncio.run(main())
