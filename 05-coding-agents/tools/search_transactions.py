"""Searches/lists transactions, filterable by account, category, merchant, date range, and amount range.

Inputs (all optional, combined with AND semantics; no filters returns everything):
    account_id — format ACC-#####
    category_id — format CAT-##
    merchant_id — format MER-####
    start_date, end_date — YYYY-MM-DD, inclusive range
    min_amount, max_amount — signed transaction amount range (negative = money out)
Returns:
    {"transactions": [...], "count": int}
"""
from tools._data import load

TOOL_SPEC = {
    "toolSpec": {
        "name": "search_transactions",
        "description": "Search transactions filtered by any combination of account, category, merchant, date range, or amount range.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": ["string", "null"],
                        "pattern": "^ACC-[0-9]{5}$",
                        "description": "Optional filter: return only transactions belonging to this account. Format is 'ACC-' followed by 5 digits, e.g. 'ACC-00123'. Omit or set to null to not filter by account.",
                    },
                    "category_id": {
                        "type": ["string", "null"],
                        "pattern": "^CAT-[0-9]{2}$",
                        "description": "Optional filter: return only transactions in this category. Format is 'CAT-' followed by 2 digits, e.g. 'CAT-07'. Omit or set to null to not filter by category.",
                    },
                    "merchant_id": {
                        "type": ["string", "null"],
                        "pattern": "^MER-[0-9]{4}$",
                        "description": "Optional filter: return only transactions with this merchant. Format is 'MER-' followed by 4 digits, e.g. 'MER-0456'. Omit or set to null to not filter by merchant.",
                    },
                    "start_date": {
                        "type": ["string", "null"],
                        "format": "date",
                        "description": "Optional filter: earliest transaction date to include, inclusive, in YYYY-MM-DD format. Omit or set to null for no lower bound on date.",
                    },
                    "end_date": {
                        "type": ["string", "null"],
                        "format": "date",
                        "description": "Optional filter: latest transaction date to include, inclusive, in YYYY-MM-DD format. Omit or set to null for no upper bound on date.",
                    },
                    "min_amount": {
                        "type": ["number", "null"],
                        "description": "Optional filter: minimum signed transaction amount to include, inclusive. Negative values are money out, positive are money in. Omit or set to null for no lower bound.",
                    },
                    "max_amount": {
                        "type": ["number", "null"],
                        "description": "Optional filter: maximum signed transaction amount to include, inclusive. Negative values are money out, positive are money in. Omit or set to null for no upper bound.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            }
        },
    }
}


def handle(
    account_id=None,
    category_id=None,
    merchant_id=None,
    start_date=None,
    end_date=None,
    min_amount=None,
    max_amount=None,
):
    txns = load("transactions")

    def keep(t):
        if account_id is not None and t["account_id"] != account_id:
            return False
        if category_id is not None and t["category_id"] != category_id:
            return False
        if merchant_id is not None and t["merchant_id"] != merchant_id:
            return False
        if start_date is not None and t["date"] < start_date:
            return False
        if end_date is not None and t["date"] > end_date:
            return False
        if min_amount is not None and t["amount"] < min_amount:
            return False
        if max_amount is not None and t["amount"] > max_amount:
            return False
        return True

    results = [t for t in txns if keep(t)]
    return {"transactions": results, "count": len(results)}
