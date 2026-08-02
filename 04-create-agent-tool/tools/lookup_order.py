"""Look up order and account details by order ID or customer email.

Inputs:
    order_id — the order identifier (e.g. "ORD-98231"); provide this and/or email
    email    — the customer's account email (e.g. "jsmith@example.com"); provide this and/or order_id
Returns:
    dict with keys: found, order_id, status, items (list), shipping_address (dict),
    estimated_delivery, customer_email, customer_name
    — or {"found": false} when no matching record exists.
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "lookup_order",
        "description": (
            "Look up order and account details for a customer. Provide either order_id "
            "or email (or both). Returns order status, line items, shipping address, "
            "estimated delivery date, and customer name/email. Returns "
            "{\"found\": false} when no record matches."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": [],
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": (
                            "The order identifier to look up, e.g. \"ORD-98231\". "
                            "Provide this and/or email; at least one is required."
                        ),
                    },
                    "email": {
                        "type": "string",
                        "description": (
                            "The customer's account email address, e.g. \"jsmith@example.com\". "
                            "Provide this and/or order_id; at least one is required."
                        ),
                    },
                },
            }
        },
    }
}

_KNOWN_ORDERS = {"ORD-98231", "ORD-77104"}
_KNOWN_EMAILS = {"jsmith@example.com", "mlopez@example.com"}


def _mock_receiver(order_id: str = None, email: str = None) -> dict:
    """Stand-in for the real order-management backend.

    No real services yet, so this returns a plausible mock record. Swap the
    body here when a real API or DB call is ready.
    """
    if order_id and order_id not in _KNOWN_ORDERS:
        return {"found": False}
    if email and not order_id and email not in _KNOWN_EMAILS:
        return {"found": False}

    resolved_id = order_id or "ORD-77104"
    return {
        "found": True,
        "order_id": resolved_id,
        "status": "in_transit",
        "customer_name": "Jane Smith",
        "customer_email": email or "jsmith@example.com",
        "items": [
            {"sku": "WH-XR5000", "name": "XR-5000 Wireless Headset", "qty": 1, "unit_price": 149.99}
        ],
        "shipping_address": {
            "line1": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip": "62701",
            "country": "US",
        },
        "estimated_delivery": "2026-08-10",
    }


def handle(order_id: str = None, email: str = None) -> dict:
    """Run the tool. Requires at least one of order_id or email."""
    if not order_id and not email:
        raise ValueError("At least one of order_id or email must be provided")
    return _mock_receiver(order_id=order_id, email=email)
