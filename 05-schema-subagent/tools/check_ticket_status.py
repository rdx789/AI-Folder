"""Looks up a support ticket by ticket ID and returns its status, priority,
subject, and last update timestamp.

Inputs:
    ticket_id — the support ticket identifier, e.g. "T-5521"
Returns:
    a dict with ticket_id, status, priority, subject, and updated_at (or an
    error if the ticket isn't found)
"""
import re

TOOL_SPEC = {
    "toolSpec": {
        "name": "check_ticket_status",
        "description": "Look up a support ticket's current status, priority, subject, and last update timestamp by ticket ID.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ticket_id"],
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The unique identifier of the support ticket to look up, formatted as an uppercase letter prefix followed by a hyphen and digits, e.g. 'T-5521'. Must exactly match the ticket ID as stored in the support system, including the prefix and leading zeros if any.",
                        "pattern": "^[A-Z]+-[0-9]+$",
                        "minLength": 3,
                    },
                },
            }
        },
    }
}

_TICKET_ID_PATTERN = re.compile(r"^[A-Z]+-[0-9]+$")


def _mock_receiver(ticket_id: str) -> dict:
    """Stand-in for the real ticketing-system backend.

    Returns a plausible mock ticket record. Swap this body for a real API/DB
    call later — the shape returned here is the contract the agent expects.
    """
    return {
        "ticket_id": ticket_id,
        "status": "in_progress",
        "priority": "normal",
        "subject": "Broken blender received",
        "updated_at": "2026-08-01T14:32:00Z",
    }


def handle(**kwargs) -> dict:
    ticket_id = kwargs.get("ticket_id")
    if not ticket_id or not _TICKET_ID_PATTERN.match(ticket_id):
        return {"error": "ticket_id is required and must match the PREFIX-NUMBER format, e.g. 'T-5521'"}
    return _mock_receiver(ticket_id)
