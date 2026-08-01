"""Creates or escalates a support ticket for issues the agent can't resolve directly."""

TOOL_SPEC = {
    "toolSpec": {
        "name": "create_ticket",
        "description": (
            "Create a new support ticket, or escalate an existing issue to a "
            "human agent, when the request needs follow-up beyond what can "
            "be resolved in this conversation."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A short summary of the customer's issue.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "description": "Urgency of the issue.",
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "The customer's email address, for follow-up.",
                    },
                },
                "required": ["summary", "priority", "customer_email"],
                "additionalProperties": False,
            }
        },
    }
}


def _mock_ticketing_backend(summary: str, priority: str, customer_email: str) -> dict:
    return {
        "ticket_id": "TCK-9981",
        "summary": summary,
        "priority": priority,
        "customer_email": customer_email,
        "status": "open",
    }


def handle(summary: str, priority: str, customer_email: str) -> dict:
    return _mock_ticketing_backend(summary, priority, customer_email)
