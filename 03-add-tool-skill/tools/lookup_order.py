"""Looks up an order/account by order ID or customer email."""

TOOL_SPEC = {
    "toolSpec": {
        "name": "lookup_order",
        "description": (
            "Look up an order by order ID or customer email. Returns order "
            "status, line items, and total. Use this whenever a customer "
            "asks about a specific order or their account's recent orders."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": (
                            "The order ID to look up, e.g. 'ORD-10234'. "
                            "Provide this or customer_email."
                        ),
                    },
                    "customer_email": {
                        "type": "string",
                        "description": (
                            "The customer's email address to look up their "
                            "most recent order. Provide this or order_id."
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            }
        },
    }
}


def _mock_order_backend(order_id: str | None, customer_email: str | None) -> dict:
    return {
        "order_id": order_id or "ORD-10234",
        "customer_email": customer_email or "customer@example.com",
        "status": "shipped",
        "items": [
            {"sku": "WBH-100", "name": "Wireless Headphones", "quantity": 1, "price": 79.99},
        ],
        "total": 79.99,
    }


def handle(order_id: str = None, customer_email: str = None) -> dict:
    return _mock_order_backend(order_id, customer_email)
