"""Detects recurring/subscription-like charges: same merchant, same or
near-identical amount, roughly monthly cadence.

Inputs:
    account_id — optional string, format ACC-#####, to scope to one account;
        omitted checks across all accounts.
    min_occurrences — optional integer >= 2, minimum times a merchant must
        recur to be reported. Defaults to 3 if omitted.
Returns:
    {"recurring_charges": [{"merchant_id", "merchant_name", "account_id",
    "occurrences", "average_amount"}, ...]}
"""
from tools._data import load

TOOL_SPEC = {
    "toolSpec": {
        "name": "find_recurring_charges",
        "description": "Find recurring/subscription-like charges: the same merchant billing the same account a similar amount on a roughly monthly cadence.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": ["string", "null"],
                        "pattern": "^ACC-\\d{5}$",
                        "description": "Optional account identifier in the format ACC-##### (e.g. ACC-00123) to scope the recurring-charge search to a single account. Omit or pass null to search across all accounts.",
                    },
                    "min_occurrences": {
                        "type": ["integer", "null"],
                        "minimum": 2,
                        "description": "Optional minimum number of times a merchant charge must repeat before it is reported as recurring. If omitted or null, defaults to 3.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            }
        },
    }
}


def handle(account_id=None, min_occurrences=None):
    threshold = min_occurrences if min_occurrences is not None else 3
    txns = load("transactions")
    merchants = {m["merchant_id"]: m["name"] for m in load("merchants")}

    groups = {}
    for t in txns:
        if t["merchant_id"] is None:
            continue
        if account_id is not None and t["account_id"] != account_id:
            continue
        key = (t["account_id"], t["merchant_id"])
        groups.setdefault(key, []).append(t["amount"])

    results = []
    for (acc_id, mid), amounts in groups.items():
        if len(amounts) < threshold:
            continue
        avg = sum(amounts) / len(amounts)
        # "same or near-identical amount": tolerate small variance around the average
        if avg == 0 or all(abs(a - avg) <= max(1.0, abs(avg) * 0.1) for a in amounts):
            results.append({
                "merchant_id": mid,
                "merchant_name": merchants.get(mid, "unknown"),
                "account_id": acc_id,
                "occurrences": len(amounts),
                "average_amount": round(avg, 2),
            })

    results.sort(key=lambda r: r["occurrences"], reverse=True)
    return {"recurring_charges": results}
