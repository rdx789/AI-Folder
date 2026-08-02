"""The tool-use agent loop.

Send the user request + available tool specs to the model via Bedrock
Converse. If the model returns a toolUse block, run the matching tool and
feed the toolResult back; repeat until the model returns a final text
answer. Capped at MAX_TURNS so a confused model can't loop forever.
"""
from agent.client import get_client, get_model_id
from agent.tool_registry import discover_tools

MAX_TURNS = 8


def run_agent(system_prompt: str, user_prompt: str) -> str:
    client = get_client()
    model_id = get_model_id()
    tool_specs, handlers = discover_tools()

    messages = [{"role": "user", "content": [{"text": user_prompt}]}]
    tool_config = {"tools": tool_specs} if tool_specs else None

    for _ in range(MAX_TURNS):
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "system": [{"text": system_prompt}],
        }
        if tool_config:
            kwargs["toolConfig"] = tool_config

        response = client.converse(**kwargs)
        output_message = response["output"]["message"]
        messages.append(output_message)

        stop_reason = response["stopReason"]
        if stop_reason != "tool_use":
            return "\n".join(
                block["text"] for block in output_message["content"] if "text" in block
            )

        tool_results = []
        for block in output_message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            handler = handlers.get(tool_use["name"])
            if handler is None:
                content = [{"text": f"Unknown tool: {tool_use['name']}"}]
            else:
                try:
                    content = [{"json": handler(**tool_use["input"])}]
                except Exception as exc:
                    content = [{"text": f"Tool error: {exc}"}]
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": content,
                    }
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Reached the turn limit without a final answer."
