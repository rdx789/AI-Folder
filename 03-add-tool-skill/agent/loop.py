"""The tool-use loop: send messages to the model, run any requested tools,
feed results back, repeat until the model returns a final text answer."""

from agent.client import call_model
from tools.registry import dispatch, get_tool_config

MAX_TURNS = 8


def run(user_prompt: str, system_prompt: str) -> str:
    messages = [{"role": "user", "content": [{"text": user_prompt}]}]
    system = [{"text": system_prompt}]
    tool_config = get_tool_config()

    for _ in range(MAX_TURNS):
        response = call_model(messages, system, tool_config)
        output_message = response["output"]["message"]
        messages.append(output_message)

        if response["stopReason"] != "tool_use":
            return "".join(
                block["text"] for block in output_message["content"] if "text" in block
            )

        tool_results = []
        for block in output_message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            try:
                result = dispatch(tool_use["name"], tool_use["input"])
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"json": result}],
                        }
                    }
                )
            except Exception as exc:
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": str(exc)}],
                            "status": "error",
                        }
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return "Reached max turns without a final answer."
