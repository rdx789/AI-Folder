"""Tool-use agent loop.

Sends the user message to the model, executes any tool calls it requests,
feeds results back, and repeats until the model produces a final text answer
or the turn cap is hit.
"""
import importlib
from pathlib import Path

from agent.client import call_model, get_client, get_model_id
from prompts.system_prompt import SYSTEM_PROMPT

MAX_TURNS = 10


def _discover_tools() -> dict:
    """Import every tools/*.py that exports TOOL_SPEC and handle().

    Returns a dict mapping tool name -> (tool_spec_entry, handle_fn).
    A new file in tools/ is all it takes to register a new capability.
    """
    tools_dir = Path(__file__).parent.parent / "tools"
    registry: dict = {}
    for path in sorted(tools_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"tools.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, "TOOL_SPEC") and hasattr(mod, "handle"):
            name = mod.TOOL_SPEC["toolSpec"]["name"]
            registry[name] = (mod.TOOL_SPEC, mod.handle)
    return registry


def run(user_message: str) -> str:
    """Run the agent loop for one user turn. Returns the final text response."""
    client = get_client()
    model_id = get_model_id()

    tool_registry = _discover_tools()
    tool_specs = [spec for spec, _ in tool_registry.values()]

    messages = [{"role": "user", "content": [{"text": user_message}]}]

    for _ in range(MAX_TURNS):
        response = call_model(client, model_id, SYSTEM_PROMPT, messages, tool_specs)
        stop_reason = response["stopReason"]
        output_message = response["output"]["message"]
        messages.append(output_message)

        if stop_reason == "end_turn":
            for block in output_message["content"]:
                if "text" in block:
                    return block["text"]
            return ""

        if stop_reason == "tool_use":
            tool_results = []
            for block in output_message["content"]:
                if "toolUse" not in block:
                    continue
                tool_use = block["toolUse"]
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]

                if tool_name in tool_registry:
                    _, handle = tool_registry[tool_name]
                    try:
                        result = handle(**tool_input)
                        result_content = [{"json": result}]
                        status = "success"
                    except Exception as exc:
                        result_content = [{"text": str(exc)}]
                        status = "error"
                else:
                    result_content = [{"text": f"Unknown tool: {tool_name}"}]
                    status = "error"

                tool_results.append({
                    "toolUseId": tool_use["toolUseId"],
                    "content": result_content,
                    "status": status,
                })

            messages.append({
                "role": "user",
                "content": [{"toolResult": tr} for tr in tool_results],
            })
        else:
            for block in output_message["content"]:
                if "text" in block:
                    return block["text"]
            return f"[stopped: {stop_reason}]"

    return "[turn cap reached — agent did not produce a final answer]"
