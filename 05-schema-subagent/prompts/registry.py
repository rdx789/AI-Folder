"""Registry mapping a short scenario name to its sample prompt module.

Add a new scenario by dropping a prompts/<name>.py file (exposing a PROMPT
string) and registering it here.
"""
from prompts.angry_refund_request import PROMPT as angry_refund_request
from prompts.billing_dispute import PROMPT as billing_dispute
from prompts.order_status_check import PROMPT as order_status_check
from prompts.password_reset_howto import PROMPT as password_reset_howto
from prompts.product_feature_question import PROMPT as product_feature_question

SCENARIOS = {
    "angry_refund_request": angry_refund_request,
    "order_status_check": order_status_check,
    "password_reset_howto": password_reset_howto,
    "billing_dispute": billing_dispute,
    "product_feature_question": product_feature_question,
}


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS)


def get_scenario(name: str) -> str:
    return SCENARIOS[name]
