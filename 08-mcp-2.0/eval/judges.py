"""Three LLM-as-judge metrics for tool-using agents — the OUTCOME half of the eval.

The counterpart to checks.py. Those metrics score the trajectory with plain asserts;
these score the things you can't assert — how the final ANSWER relates to the tool
results and the question. Each is one Nova call, built the same way as Lesson 7's
judges:
  - a SYSTEM prompt holds the RUBRIC (the judge's role and what 0.0 vs 1.0 mean);
  - the user message is a few LABELLED sections (question / tool results / answer);
  - the score returns through a FORCED tool call (submit_score), so we always get a
    clean {score, reason} instead of parsing a number out of prose.

The three metrics form a chain with the process checks — a break upstream caps
everything downstream:
    selection → arguments → result_utilization → result_faithfulness → answer_relevance
  - result_utilization — is the answer actually BUILT from the tool results, or did
                         the model ignore them and answer from its own knowledge?
  - result_faithfulness — is every claim in the answer SUPPORTED by the tool results
                         (no invented ids, no numbers the tools never returned)?
  - answer_relevance    — does the answer ADDRESS the question that was asked
                         (asking to disambiguate an ambiguous request counts)?

utilization vs faithfulness is a real distinction, not a duplicate: utilization asks
whether the answer *leans on* the results at all; faithfulness asks whether it stays
*consistent with* them. An answer can use the results and still contradict them.
"""
import os

import boto3
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

# One shared scoring tool for all three judges. Forcing it (toolChoice) guarantees a
# structured {score, reason} — no free-text parsing. What the 0.0-1.0 scale MEANS is
# defined per-metric in each rubric (the system prompt), not here.
_SCORE_TOOL = {
    "toolSpec": {
        "name": "submit_score",
        "description": "Submit the evaluation score for the answer under review.",
        "inputSchema": {"json": {
            "type": "object", "additionalProperties": False, "required": ["score", "reason"],
            "properties": {
                "score": {"type": "number", "description": "A value from 0.0 (worst) to 1.0 (best), per the rubric."},
                "reason": {"type": "string", "description": "One short sentence justifying the score."},
            },
        }},
    }
}


def _run_judge(rubric: str, sections: list[tuple[str, str]]) -> tuple[float, str]:
    """One judge call. `rubric` becomes the system prompt; `sections` are the labelled
    blocks of the user message ([(label, text), ...]); the score returns through the
    forced submit_score tool as (score, reason)."""
    user_message = "\n\n".join(f"### {label}\n{body}" for label, body in sections)
    resp = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": rubric}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        toolConfig={"tools": [_SCORE_TOOL], "toolChoice": {"tool": {"name": "submit_score"}}},
        inferenceConfig={"temperature": 0.0},   # a judge should score the same input the same way
    )
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            result = block["toolUse"]["input"]
            score = max(0.0, min(1.0, float(result.get("score", 0.0))))
            return score, result.get("reason", "")
    return 0.0, ""


# --- the rubrics (each defines its own 0.0-1.0 scale) --------------------------

UTILIZATION_RUBRIC = (
    "You are a strict evaluator scoring TOOL RESULT UTILIZATION — whether the ANSWER "
    "was actually built from the TOOL RESULTS the agent retrieved, rather than from "
    "the model's own prior knowledge.\n"
    "  1.0 = the answer's substantive claims come from the tool results; the tools "
    "clearly drove the answer.\n"
    "  0.0 = the answer ignores the tool results — it could have been written without "
    "ever calling them, or it sidesteps what they returned.\n"
    "Judge only how much the answer DRAWS ON the tool results — not whether it is "
    "correct or complete."
)

FAITHFULNESS_RUBRIC = (
    "You are a strict evaluator scoring FAITHFULNESS — whether the ANSWER is grounded "
    "in the TOOL RESULTS.\n"
    "  1.0 = every factual claim in the answer is supported by the tool results.\n"
    "  0.0 = the answer states facts the tool results do not support — a fabricated id, "
    "an invented number, a claim that contradicts what the tools returned.\n"
    "Judge support by the TOOL RESULTS only; ignore whether a claim happens to be true "
    "in the real world."
)

RELEVANCE_RUBRIC = (
    "You are a strict evaluator scoring ANSWER RELEVANCE — whether the ANSWER addresses "
    "the QUESTION the user actually asked.\n"
    "  1.0 = the answer directly and appropriately responds to the question. When the "
    "question is ambiguous or under-specified, appropriately asking for the missing "
    "detail (e.g. which of two people is meant) is a relevant response and scores high.\n"
    "  0.0 = the answer is off-topic, evasive, or answers a different question.\n"
    "Judge relevance to the QUESTION only — not factual correctness or grounding."
)


# --- the three judges: assemble sections, run the rubric -----------------------

def result_utilization(question: str, tool_results: str, answer: str) -> tuple[float, str]:
    return _run_judge(UTILIZATION_RUBRIC, [
        ("QUESTION", question),
        ("TOOL RESULTS", tool_results),
        ("ANSWER", answer),
    ])


def result_faithfulness(question: str, tool_results: str, answer: str) -> tuple[float, str]:
    return _run_judge(FAITHFULNESS_RUBRIC, [
        ("QUESTION", question),
        ("TOOL RESULTS", tool_results),
        ("ANSWER", answer),
    ])


def answer_relevance(question: str, answer: str) -> tuple[float, str]:
    return _run_judge(RELEVANCE_RUBRIC, [
        ("QUESTION", question),
        ("ANSWER", answer),
    ])
