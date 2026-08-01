"""Checks the status of a support ticket by ticket ID."""

TOOL_SPEC = {
    "toolSpec": {
        "name": "check_ticket_status",
        "description": (
            "Check the current status, priority, and last update of an "
            "existing support ticket by its ticket ID."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The support ticket ID to check, e.g. 'TCK-5521'.",
                    },
                },
                "required": ["ticket_id"],
                "additionalProperties": False,
            }
        },
    }
}


def _mock_ticket_backend(ticket_id: str) -> dict:
    return {
        "ticket_id": ticket_id,
        "status": "in_progress",
        "priority": "normal",
        "assigned_to": "support-tier-1",
        "last_update": "Agent requested additional info from customer.",
    }


def handle(ticket_id: str) -> dict:
    return _mock_ticket_backend(ticket_id)
