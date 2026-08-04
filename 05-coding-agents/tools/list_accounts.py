"""Lists the user's accounts, optionally filtered by account_type.

Inputs:
    account_type — optional filter: "checking", "savings", or "credit_card".
        Omit or pass null to return accounts of all types.
Returns:
    {"accounts": [{"account_id", "owner_name", "account_type", "opened_at",
    "balance"}, ...]}
"""
from tools._data import load

TOOL_SPEC = {
    "toolSpec": {
        "name": "list_accounts",
        "description": "List the user's accounts, optionally filtered by account type.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": ["string", "null"],
                        "enum": ["checking", "savings", "credit_card", None],
                        "description": "Optional filter to restrict results to accounts of a single type. One of: 'checking', 'savings', 'credit_card'. Omit or pass null to return accounts of all types.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            }
        },
    }
}


def handle(account_type=None):
    accounts = load("accounts")
    if account_type is not None:
        accounts = [a for a in accounts if a["account_type"] == account_type]
    return {"accounts": accounts}
