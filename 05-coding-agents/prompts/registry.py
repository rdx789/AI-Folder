"""Registry of sample scenarios: short name -> user prompt module."""
from prompts.dining_last_month import PROMPT as dining_last_month
from prompts.groceries_quarter_comparison import PROMPT as groceries_quarter_comparison
from prompts.most_active_account import PROMPT as most_active_account
from prompts.recurring_subscriptions import PROMPT as recurring_subscriptions
from prompts.unusual_large_transactions import PROMPT as unusual_large_transactions

SCENARIOS = {
    "dining_last_month": dining_last_month,
    "recurring_subscriptions": recurring_subscriptions,
    "most_active_account": most_active_account,
    "groceries_quarter_comparison": groceries_quarter_comparison,
    "unusual_large_transactions": unusual_large_transactions,
}


def list_scenarios():
    return list(SCENARIOS.keys())


def get_scenario(name):
    return SCENARIOS[name]
