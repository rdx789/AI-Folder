"""Registry mapping a short scenario name to its sample prompt module.

Add a scenario by dropping a new prompts/<name>.py (with DESCRIPTION and
PROMPT module-level strings) and adding it to SCENARIOS below.
"""
from prompts import (
    angry_refund_request,
    billing_dispute,
    order_status_check,
    password_reset_howto,
    product_feature_question,
)

SCENARIOS = {
    "angry_refund_request": angry_refund_request,
    "order_status_check": order_status_check,
    "password_reset_howto": password_reset_howto,
    "billing_dispute": billing_dispute,
    "product_feature_question": product_feature_question,
}


def list_scenarios() -> dict:
    """Return {name: description} for every registered scenario."""
    return {name: mod.DESCRIPTION for name, mod in SCENARIOS.items()}


def get_prompt(name: str) -> str:
    """Return the sample PROMPT text for a registered scenario name."""
    return SCENARIOS[name].PROMPT
