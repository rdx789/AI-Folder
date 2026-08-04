"""The tool-use loop: send request + tool specs, run any toolUse, repeat."""
from agent.client import get_client, get_model_id
from agent.registry import discover_tools

MAX_TURNS = 8


def run_agent(system_prompt, user_prompt, max_turns=MAX_TURNS, on_tool_call=None):
    """Run the agent to completion. Returns (final_text, trace).

    trace is a list of {"tool", "input", "result"} or {"tool", "input", "error"}
    dicts, one per tool call, in call order.
    """
    client = get_client()
    model_id = get_model_id()
    tool_specs, handlers = discover_tools()
    tool_config = {"tools": tool_specs} if tool_specs else None

    messages = [{"role": "user", "content": [{"text": user_prompt}]}]
    trace = []

    for _ in range(max_turns):
        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "system": [{"text": system_prompt}],
        }
        if tool_config:
            kwargs["toolConfig"] = tool_config

        response = client.converse(**kwargs)
        message = response["output"]["message"]
        messages.append(message)

        tool_uses = [block["toolUse"] for block in message["content"] if "toolUse" in block]
        if not tool_uses:
            final_text = "".join(block["text"] for block in message["content"] if "text" in block)
            return final_text, trace

        tool_results = []
        for tool_use in tool_uses:
            name = tool_use["name"]
            tool_input = tool_use.get("input", {})
            handler = handlers.get(name)
            try:
                if handler is None:
                    raise ValueError(f"no such tool: {name}")
                result = handler(**tool_input)
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": result}],
                    }
                })
                trace.append({"tool": name, "input": tool_input, "result": result})
            except Exception as e:  # a bad tool input or a broken handler shouldn't crash the loop
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": str(e)}],
                        "status": "error",
                    }
                })
                trace.append({"tool": name, "input": tool_input, "error": str(e)})

            if on_tool_call:
                on_tool_call(trace[-1])

        messages.append({"role": "user", "content": tool_results})

    return "I reached my turn limit without finishing. Please try a narrower question.", trace
