"""Looks up an order by order ID and returns its status, items, total, and
shipping info. Used to answer "where is my order" and general order lookups.

Inputs:
    order_id — the order identifier, e.g. "A1029" or "#B4471"
    customer_email — optional, used to verify the order belongs to the customer
Returns:
    a dict with order_id, status, items, total, and shipping info (or an
    error if the order isn't found)
"""
import re

TOOL_SPEC = {
    "toolSpec": {
        "name": "lookup_order",
        "description": "Look up an order by order ID and return its status, items, total, and shipping info. Call this to answer 'where is my order' or general order questions.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["order_id", "customer_email"],
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier to look up, e.g. 'A1029' or '#B4471'. A short alphanumeric code, optionally prefixed with '#' and/or a leading letter. Pass the identifier as given by the customer, including any leading '#' if present.",
                        "pattern": "^#?[A-Za-z0-9]{3,12}$",
                    },
                    "customer_email": {
                        "type": ["string", "null"],
                        "description": "The customer's email address, used to verify that the order belongs to them before returning details. Omit or set to null if the customer's email is not known or was not provided.",
                    },
                },
            }
        },
    }
}

_ORDER_ID_PATTERN = re.compile(r"^#?[A-Za-z0-9]{3,12}$")


def _mock_receiver(order_id: str, customer_email: str | None) -> dict:
    """Stand-in for the real order-management backend.

    Returns a plausible mock order record. Swap this body for a real API/DB
    call later — the shape returned here is the contract the agent expects.
    """
    return {
        "order_id": order_id,
        "status": "shipped",
        "items": [{"sku": "AB-200", "name": "Aura Blender Pro", "quantity": 1}],
        "total": 79.99,
        "shipping": {"carrier": "UPS", "tracking_number": "1Z999AA10123456784", "eta": "2026-08-05"},
        "customer_email": customer_email,
    }


def handle(**kwargs) -> dict:
    order_id = kwargs.get("order_id")
    if not order_id or not _ORDER_ID_PATTERN.match(order_id):
        return {"error": "order_id is required and must be a valid order identifier"}
    return _mock_receiver(order_id, kwargs.get("customer_email"))
