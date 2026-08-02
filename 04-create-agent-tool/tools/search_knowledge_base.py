"""Search the help-center knowledge base for articles matching a query.

Inputs:
    query    — free-text search query (e.g. "how do I reset my password")
    max_results — max number of articles to return (1-10, default 3)
Returns:
    dict with key: results, a list of {title, url, snippet} — empty list if
    nothing matches.
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "search_knowledge_base",
        "description": (
            "Search the help-center knowledge base for articles relevant to a "
            "customer's question, e.g. password resets, shipping policy, return "
            "policy. Use this before asking the customer to repeat information "
            "that's already documented. Returns a ranked list of matching articles."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query, e.g. \"how do I reset my password\".",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of articles to return. Must be between 1 and 10. Defaults to 3 if omitted.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
            }
        },
    }
}

_ARTICLES = [
    {
        "title": "How to reset your password",
        "url": "https://help.example.com/articles/password-reset",
        "keywords": {"password", "reset", "login", "sign", "in"},
        "snippet": "Go to Account Settings > Security > Reset Password, and follow the emailed link.",
    },
    {
        "title": "Shipping times and tracking your order",
        "url": "https://help.example.com/articles/shipping-tracking",
        "keywords": {"shipping", "order", "tracking", "delivery", "ship"},
        "snippet": "Standard shipping takes 5-7 business days. Track your order from the Orders page.",
    },
    {
        "title": "Return and refund policy",
        "url": "https://help.example.com/articles/returns-refunds",
        "keywords": {"refund", "return", "money", "back", "damaged"},
        "snippet": "Most items can be returned within 30 days of delivery for a full refund.",
    },
    {
        "title": "Understanding your billing statement",
        "url": "https://help.example.com/articles/billing",
        "keywords": {"billing", "charge", "subscription", "invoice", "payment"},
        "snippet": "Charges appear as 'EXAMPLE*ORDER' on your statement. Subscriptions renew monthly.",
    },
]


def _mock_receiver(query: str, max_results: int) -> dict:
    """Stand-in for the real knowledge-base search backend.

    No real services yet, so this does simple keyword overlap scoring against
    a small fixture set. Swap the body here when a real search API is ready.
    """
    query_words = set(query.lower().split())
    scored = []
    for article in _ARTICLES:
        overlap = len(query_words & article["keywords"])
        if overlap > 0:
            scored.append((overlap, article))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [
        {"title": a["title"], "url": a["url"], "snippet": a["snippet"]}
        for _, a in scored[:max_results]
    ]
    return {"results": results}


def handle(query: str, max_results: int = 3) -> dict:
    """Run the tool. query is required; max_results defaults to 3."""
    if not query:
        raise ValueError("query must be a non-empty string")
    if not (1 <= max_results <= 10):
        raise ValueError("max_results must be between 1 and 10")
    return _mock_receiver(query=query, max_results=max_results)
