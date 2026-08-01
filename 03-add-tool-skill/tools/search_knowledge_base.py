"""Searches the help-center knowledge base for articles matching a query."""

TOOL_SPEC = {
    "toolSpec": {
        "name": "search_knowledge_base",
        "description": (
            "Search the help-center knowledge base for articles relevant to "
            "a customer's question, e.g. how-to guides or policy pages. Use "
            "this before answering how-to or policy questions from memory."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'reset password' or 'return policy'.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        },
    }
}


def _mock_kb_backend(query: str) -> dict:
    return {
        "query": query,
        "results": [
            {
                "title": f"How to: {query}",
                "url": f"https://help.example.com/articles/{query.replace(' ', '-').lower()}",
                "snippet": f"Step-by-step guide covering '{query}' and related account settings.",
            }
        ],
    }


def handle(query: str) -> dict:
    return _mock_kb_backend(query)
