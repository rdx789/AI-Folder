"""Totals spending (money out) by category over an optional date range and account.

Inputs:
    start_date, end_date — optional YYYY-MM-DD, inclusive range; omitted means
        no bound on that side.
    account_id — optional string, format ACC-#####, to scope to one account;
        omitted means all accounts.
Returns:
    {"spending_by_category": [{"category_id", "category_name", "total_spent"}, ...]}
    sorted by total_spent descending (most negative first, i.e. biggest spend).
    Only spend transactions (negative amount) are counted; income/transfers-in
    are excluded.
"""
from tools._data import load

TOOL_SPEC = {
    "toolSpec": {
        "name": "get_spending_by_category",
        "description": "Get total spending grouped by category, optionally scoped to a date range and/or one account.",
        "inputSchema": {
            "json": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_date": {
                        "type": ["string", "null"],
                        "description": "Inclusive start of the date range to include, as YYYY-MM-DD. If omitted (null), spending is aggregated from the beginning of the available transaction history.",
                    },
                    "end_date": {
                        "type": ["string", "null"],
                        "description": "Inclusive end of the date range to include, as YYYY-MM-DD. If omitted (null), spending is aggregated through the most recent available transaction.",
                    },
                    "account_id": {
                        "type": ["string", "null"],
                        "pattern": "^ACC-\\d{5}$",
                        "description": "Scope the spending summary to a single account, formatted as ACC-##### (e.g. ACC-00123). If omitted (null), spending is aggregated across all accounts.",
                    },
                },
                "required": [],
            }
        },
    }
}


def handle(start_date=None, end_date=None, account_id=None):
    txns = load("transactions")
    categories = {c["category_id"]: c["name"] for c in load("categories")}

    totals = {}
    for t in txns:
        if t["amount"] >= 0:
            continue
        if account_id is not None and t["account_id"] != account_id:
            continue
        if start_date is not None and t["date"] < start_date:
            continue
        if end_date is not None and t["date"] > end_date:
            continue
        totals[t["category_id"]] = totals.get(t["category_id"], 0.0) + t["amount"]

    result = [
        {
            "category_id": cid,
            "category_name": categories.get(cid, "unknown"),
            "total_spent": round(total, 2),
        }
        for cid, total in totals.items()
    ]
    result.sort(key=lambda r: r["total_spent"])
    return {"spending_by_category": result}
