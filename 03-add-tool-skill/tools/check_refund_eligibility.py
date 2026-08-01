"""Checks whether an order is eligible for refund or return."""

TOOL_SPEC = {
    "toolSpec": {
        "name": "check_refund_eligibility",
        "description": (
            "Check whether a given order is eligible for a refund or return, "
            "based on purchase date and item condition policy. Use this "
            "before promising or denying a refund."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to check for refund/return eligibility.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The customer's stated reason for the refund/return request.",
                    },
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False,
            }
        },
    }
}


def _mock_refund_backend(order_id: str, reason: str) -> dict:
    return {
        "order_id": order_id,
        "reason": reason,
        "eligible": True,
        "policy_window_days": 30,
        "days_since_purchase": 12,
        "refund_method": "original_payment_method",
    }


def handle(order_id: str, reason: str) -> dict:
    return _mock_refund_backend(order_id, reason)
