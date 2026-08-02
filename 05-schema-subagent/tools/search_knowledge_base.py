"""Searches the help-center knowledge base by free-text query, with optional
category filtering and result limiting.

Inputs:
    query — free-text search string
    category — optional, restrict to one of a fixed set of categories
    max_results — optional, how many results to return (1-10, default 3)
Returns:
    a dict with query and a list of matching articles (title, article_id, url)
"""

TOOL_SPEC = {
    "toolSpec": {
        "name": "search_knowledge_base",
        "description": "Search the help-center knowledge base by free-text query, with optional category filter, to find relevant help articles.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query", "category", "max_results"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The free-text search string used to find matching help-center articles, e.g. a customer's question or keywords describing their issue (such as 'how do I reset my password' or 'refund policy'). Must be non-empty.",
                        "minLength": 1,
                    },
                    "category": {
                        "type": ["string", "null"],
                        "description": "Restricts the search to a specific knowledge base category. Set to null or omit to search across all categories.",
                        "enum": ["account", "billing", "shipping", "returns", "product", "technical", None],
                    },
                    "max_results": {
                        "type": ["integer", "null"],
                        "description": "The maximum number of matching articles to return. If null or omitted, the tool defaults to returning about 3-5 results.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
            }
        },
    }
}

_ARTICLES = [
    {"article_id": "KB-001", "title": "How to reset your password", "category": "account"},
    {"article_id": "KB-002", "title": "Tracking your shipment", "category": "shipping"},
    {"article_id": "KB-003", "title": "Return and refund policy", "category": "returns"},
    {"article_id": "KB-004", "title": "Understanding your billing statement", "category": "billing"},
    {"article_id": "KB-005", "title": "Product warranty coverage", "category": "product"},
]


def _mock_receiver(query: str, category: str | None, max_results: int | None) -> dict:
    """Stand-in for the real knowledge-base search backend.

    Returns a plausible mock set of matching articles. Swap this body for a
    real search API call later — the shape returned here is the contract the
    agent expects.
    """
    limit = max_results or 3
    results = [a for a in _ARTICLES if category is None or a["category"] == category]
    results = results[:limit]
    return {
        "query": query,
        "results": [
            {**a, "url": f"https://help.example.com/articles/{a['article_id']}"} for a in results
        ],
    }


def handle(**kwargs) -> dict:
    query = kwargs.get("query")
    if not query:
        return {"error": "query is required"}
    max_results = kwargs.get("max_results")
    if max_results is not None and not (1 <= max_results <= 10):
        return {"error": "max_results must be between 1 and 10"}
    return _mock_receiver(query, kwargs.get("category"), max_results)
