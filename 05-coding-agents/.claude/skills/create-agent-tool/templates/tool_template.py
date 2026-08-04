"""<one line: what this tool does and when the agent should call it>.

Inputs:
    <field> — <what it is>
Returns:
    <the shape of the dict this tool returns>
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "<name>",
        "description": "<when the model should call this tool>",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": [],  # list EVERY required field name
                "properties": {
                    # "field": {"type": "string", "description": "specific, with examples/allowed values"},
                },
            }
        },
    }
}


def _mock_receiver(**kwargs) -> dict:
    """Stand-in for the real backend — returns a simple, plausible mock output.

    We have no real services yet, so the tool runs on this. It's the seam where
    a real call (HTTP, DB, SDK) plugs in later — swap the body, keep the shape.
    """
    return {}  # TODO: return a small, plausible mock dict for this tool


def handle(**kwargs) -> dict:
    """Run the tool. For now it just calls the mock receiver."""
    return _mock_receiver(**kwargs)
