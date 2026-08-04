"""Looks up the current balance and basic info for one account.

Inputs:
    account_id — required string, format ACC-##### (e.g. ACC-00123).
Returns:
    {"account_id", "owner_name", "account_type", "balance"} or
    {"error": "..."} if the account_id doesn't resolve.
"""
from tools._data import load

TOOL_SPEC = {
    "toolSpec": {
        "name": "get_account_balance",
        "description": "Get the current balance and basic info for one specific account by ID.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "The unique identifier of the account to look up, formatted as 'ACC-' followed by five digits (e.g., 'ACC-00123'). Must match an existing account.",
                        "pattern": "^ACC-\\d{5}$",
                    }
                },
                "required": ["account_id"],
                "additionalProperties": False,
            }
        },
    }
}


def handle(account_id):
    for acc in load("accounts"):
        if acc["account_id"] == account_id:
            return {
                "account_id": acc["account_id"],
                "owner_name": acc["owner_name"],
                "account_type": acc["account_type"],
                "balance": acc["balance"],
            }
    return {"error": f"no account found with account_id {account_id!r}"}
