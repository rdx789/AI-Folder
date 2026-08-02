"""Check the status of an existing support ticket by ticket ID.

Inputs:
    ticket_id — the support ticket identifier (e.g. "TCK-4471")
Returns:
    dict with keys: found, ticket_id, status, subject, priority, created_at,
    last_updated, assigned_to
    — or {"found": false} when no matching ticket exists.
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "check_ticket_status",
        "description": (
            "Check the current status of an existing support ticket. Returns the "
            "ticket's status (e.g. open, in_progress, resolved, closed), subject, "
            "priority, and timestamps. Returns {\"found\": false} when the ticket_id "
            "does not match any known ticket."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ticket_id"],
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": (
                            "The support ticket identifier to look up, e.g. \"TCK-4471\". "
                            "Always in the form TCK-<digits>."
                        ),
                    },
                },
            }
        },
    }
}

_KNOWN_TICKETS = {
    "TCK-4471": {
        "status": "in_progress",
        "subject": "Damaged blender received",
        "priority": "high",
        "created_at": "2026-07-28",
        "last_updated": "2026-07-30",
        "assigned_to": "support-tier-2",
    },
    "TCK-1029": {
        "status": "resolved",
        "subject": "Password reset email not arriving",
        "priority": "medium",
        "created_at": "2026-07-15",
        "last_updated": "2026-07-16",
        "assigned_to": "support-tier-1",
    },
}


def _mock_receiver(ticket_id: str) -> dict:
    """Stand-in for the real ticketing-system backend.

    No real services yet, so this returns a plausible mock record for known
    ticket IDs. Swap the body here when a real API or DB call is ready.
    """
    record = _KNOWN_TICKETS.get(ticket_id)
    if record is None:
        return {"found": False}
    return {"found": True, "ticket_id": ticket_id, **record}


def handle(ticket_id: str) -> dict:
    """Run the tool. ticket_id is required."""
    if not ticket_id:
        raise ValueError("ticket_id must be a non-empty string")
    return _mock_receiver(ticket_id=ticket_id)
