"""Create a new support ticket for an issue that needs follow-up.

Inputs:
    subject     — short summary of the issue (e.g. "Damaged item received")
    description — full description of the customer's issue
    priority    — one of "low", "medium", "high", "urgent"
    customer_email — the customer's account email
Returns:
    dict with keys: ticket_id, status, subject, priority, created_at
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "create_ticket",
        "description": (
            "Create a new support ticket when an issue needs to be tracked or "
            "escalated to a human agent. Returns the newly created ticket_id and "
            "its initial status."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "description", "priority", "customer_email"],
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Short summary of the issue, e.g. \"Damaged item received\".",
                    },
                    "description": {
                        "type": "string",
                        "description": "Full description of the customer's issue, including any order/account IDs mentioned.",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Urgency of the ticket.",
                        "enum": ["low", "medium", "high", "urgent"],
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "The customer's account email address, e.g. \"jsmith@example.com\".",
                    },
                },
            }
        },
    }
}

_next_id = 5000


def _mock_receiver(subject: str, description: str, priority: str, customer_email: str) -> dict:
    """Stand-in for the real ticketing-system backend.

    No real services yet, so this fabricates a new ticket ID and returns it as
    freshly created. Swap the body here when a real API or DB call is ready.
    """
    global _next_id
    _next_id += 1
    return {
        "ticket_id": f"TCK-{_next_id}",
        "status": "open",
        "subject": subject,
        "priority": priority,
        "created_at": "2026-08-02",
    }


def handle(subject: str, description: str, priority: str, customer_email: str) -> dict:
    """Run the tool. All fields are required."""
    if priority not in {"low", "medium", "high", "urgent"}:
        raise ValueError(f"Invalid priority: {priority!r}")
    if not subject or not description or not customer_email:
        raise ValueError("subject, description, and customer_email must be non-empty")
    return _mock_receiver(
        subject=subject, description=description, priority=priority, customer_email=customer_email
    )
