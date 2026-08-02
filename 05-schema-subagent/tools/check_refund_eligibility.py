"""Checks whether an order, or a specific item within it, is eligible for a
refund or return.

Inputs:
    order_id — the order identifier, e.g. "A1029"
    item_sku — optional, a specific item's SKU to check instead of the whole order
    reason — optional, one of a fixed set of return reasons
Returns:
    a dict with eligible (bool), reason_code, and a human-readable explanation
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "check_refund_eligibility",
        "description": "Check whether an order (or a specific item within it) is eligible for a refund or return, given an order ID and optional reason.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["order_id", "item_sku", "reason"],
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier to check refund/return eligibility for, e.g. 'A1029'. Must match the order ID as recorded in the order system.",
                    },
                    "item_sku": {
                        "type": ["string", "null"],
                        "description": "The SKU of a specific item within the order to check eligibility for, e.g. 'SKU-4821'. Provide this only if the customer wants to return or refund a single item rather than the entire order. Set to null or omit to check eligibility for the whole order.",
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "The customer's stated reason for the return or refund request, used to apply reason-specific eligibility rules (e.g. defective items may have different return windows than 'no longer needed'). Set to null if the customer has not given a reason.",
                        "enum": ["defective", "wrong_item", "no_longer_needed", "damaged_in_shipping", "other", None],
                    },
                },
            }
        },
    }
}


def _mock_receiver(order_id: str, item_sku: str | None, reason: str | None) -> dict:
    """Stand-in for the real order/returns backend.

    Returns a plausible mock eligibility decision. Swap this body for a real
    API/DB call later — the shape returned here is the contract the agent
    expects.
    """
    eligible = reason != "no_longer_needed"
    return {
        "order_id": order_id,
        "item_sku": item_sku,
        "eligible": eligible,
        "reason_code": reason or "unspecified",
        "explanation": (
            "Eligible for a full refund within the 30-day return window."
            if eligible
            else "Not eligible: 'no longer needed' returns are only accepted within 14 days of delivery."
        ),
    }


def handle(**kwargs) -> dict:
    order_id = kwargs.get("order_id")
    if not order_id:
        return {"error": "order_id is required"}
    reason = kwargs.get("reason")
    valid_reasons = ("defective", "wrong_item", "no_longer_needed", "damaged_in_shipping", "other")
    if reason is not None and reason not in valid_reasons:
        return {"error": f"reason must be one of: {', '.join(valid_reasons)}"}
    return _mock_receiver(order_id, kwargs.get("item_sku"), reason)
