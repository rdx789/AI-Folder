"""Creates or escalates a support ticket for a customer issue that needs
human follow-up.

Inputs:
    customer_email — the customer's email
    subject — short summary of the issue
    description — full description of the issue/conversation context
    priority — optional, one of "low", "normal", "high", "urgent"
    escalate — optional bool, whether to immediately escalate to a human
Returns:
    a dict with the new ticket_id, status, and echoed input fields
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "create_support_ticket",
        "description": "Create or escalate a support ticket for a customer issue that needs human follow-up. Call this when the agent cannot resolve the issue itself.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["customer_email", "subject", "description", "priority", "escalate"],
                "properties": {
                    "customer_email": {
                        "type": "string",
                        "description": "The email address of the customer reporting the issue, used to identify them and to send ticket updates/confirmations.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "A short, human-readable summary of the issue (e.g. 'Unable to reset password'), shown as the ticket title in the support system.",
                        "minLength": 1,
                        "maxLength": 150,
                    },
                    "description": {
                        "type": "string",
                        "description": "Full description of the issue, including relevant conversation context, steps already taken, and any details a human agent needs to pick up the case without re-asking the customer.",
                        "minLength": 1,
                    },
                    "priority": {
                        "type": ["string", "null"],
                        "description": "The urgency of the ticket. Use 'urgent' for issues causing severe business impact or requiring immediate attention, 'high' for significant impact, 'normal' for standard issues, and 'low' for minor or non-time-sensitive requests. Omit or set to null if priority cannot be determined and should default to the support system's standard setting.",
                        "enum": ["low", "normal", "high", "urgent", None],
                    },
                    "escalate": {
                        "type": ["boolean", "null"],
                        "description": "Whether this ticket should be immediately escalated to a human agent or manager rather than routed through the standard queue. Set to true when the issue is time-critical, involves a highly frustrated customer, or exceeds the bot's ability to resolve. Omit or set to null if escalation is not needed.",
                    },
                },
            }
        },
    }
}


def _mock_receiver(customer_email: str, subject: str, description: str, priority: str | None, escalate: bool | None) -> dict:
    """Stand-in for the real ticketing-system backend.

    Returns a plausible mock created-ticket record. Swap this body for a real
    API/DB call later — the shape returned here is the contract the agent
    expects.
    """
    return {
        "ticket_id": "T-9001",
        "status": "escalated" if escalate else "open",
        "customer_email": customer_email,
        "subject": subject,
        "description": description,
        "priority": priority or "normal",
    }


def handle(**kwargs) -> dict:
    for field in ("customer_email", "subject", "description"):
        if not kwargs.get(field):
            return {"error": f"{field} is required"}
    priority = kwargs.get("priority")
    if priority is not None and priority not in ("low", "normal", "high", "urgent"):
        return {"error": "priority must be one of: low, normal, high, urgent"}
    return _mock_receiver(
        kwargs["customer_email"],
        kwargs["subject"],
        kwargs["description"],
        priority,
        kwargs.get("escalate"),
    )
