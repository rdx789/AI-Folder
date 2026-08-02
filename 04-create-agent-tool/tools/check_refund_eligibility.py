"""Check whether an order is eligible for a refund or return.

Inputs:
    order_id    — the order identifier (e.g. "ORD-98231")
    reason      — one of "damaged", "wrong_item", "changed_mind", "not_as_described", "other"
Returns:
    dict with keys: eligible, order_id, reason, refund_window_days,
    days_since_delivery, policy_note
    — or {"found": false} when the order_id is unknown.
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "check_refund_eligibility",
        "description": (
            "Check whether an order qualifies for a refund or return, based on the "
            "reason given and the store's refund policy. Returns an eligible boolean "
            "plus the policy details used to decide. Returns {\"found\": false} when "
            "order_id does not match a known order."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["order_id", "reason"],
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier to check, e.g. \"ORD-98231\".",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the customer wants a refund/return.",
                        "enum": ["damaged", "wrong_item", "changed_mind", "not_as_described", "other"],
                    },
                },
            }
        },
    }
}

_KNOWN_ORDERS = {
    "ORD-98231": {"days_since_delivery": 3},
    "ORD-77104": {"days_since_delivery": 40},
}


def _mock_receiver(order_id: str, reason: str) -> dict:
    """Stand-in for the real order/refund-policy backend.

    No real services yet, so eligibility is derived from a simple mock policy:
    damaged/wrong_item/not_as_described are always eligible within 30 days;
    changed_mind/other need delivery within 14 days. Swap the body here when a
    real API or DB call is ready.
    """
    order = _KNOWN_ORDERS.get(order_id)
    if order is None:
        return {"found": False}

    days = order["days_since_delivery"]
    window = 30 if reason in {"damaged", "wrong_item", "not_as_described"} else 14
    eligible = days <= window

    return {
        "found": True,
        "eligible": eligible,
        "order_id": order_id,
        "reason": reason,
        "refund_window_days": window,
        "days_since_delivery": days,
        "policy_note": (
            f"Orders reported as '{reason}' are eligible within {window} days of delivery; "
            f"this order was delivered {days} day(s) ago."
        ),
    }


def handle(order_id: str, reason: str) -> dict:
    """Run the tool. order_id and reason are required."""
    valid_reasons = {"damaged", "wrong_item", "changed_mind", "not_as_described", "other"}
    if reason not in valid_reasons:
        raise ValueError(f"Invalid reason: {reason!r}")
    if not order_id:
        raise ValueError("order_id must be a non-empty string")
    return _mock_receiver(order_id=order_id, reason=reason)
