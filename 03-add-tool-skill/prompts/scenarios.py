"""Registry mapping a short scenario name to its sample prompt and description."""

from prompts.angry_refund_request import PROMPT as angry_refund_request_prompt
from prompts.billing_dispute import PROMPT as billing_dispute_prompt
from prompts.order_status_check import PROMPT as order_status_check_prompt
from prompts.password_reset_howto import PROMPT as password_reset_howto_prompt
from prompts.product_feature_question import PROMPT as product_feature_question_prompt

SCENARIOS = {
    "angry-refund-request": {
        "description": "Angry customer demanding a full refund for a broken item.",
        "prompt": angry_refund_request_prompt,
    },
    "order-status-check": {
        "description": "Customer asking where their order is.",
        "prompt": order_status_check_prompt,
    },
    "password-reset-howto": {
        "description": "Customer locked out of their account, needs reset steps.",
        "prompt": password_reset_howto_prompt,
    },
    "billing-dispute": {
        "description": "Customer disputing being charged twice for the same order.",
        "prompt": billing_dispute_prompt,
    },
    "product-feature-question": {
        "description": "Customer asking whether a purchased product supports a feature.",
        "prompt": product_feature_question_prompt,
    },
}


def list_scenarios():
    return {name: info["description"] for name, info in SCENARIOS.items()}


def get_prompt(name):
    return SCENARIOS[name]["prompt"]
