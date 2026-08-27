"""Subject vocabulary + Nova-based tagger shared by ingest (tags chunks) and the
pipeline's subject planner (picks which subjects a query is about).

The vocabulary was derived by reading every file under data/handbook/ and
data/manager_playbook/ and grouping their topics — it is a closed list so the
planner and the tagger always agree on the same labels.
"""
import json

from client import bedrock

MODEL_ID = None  # set lazily from env at call time to respect .env validation in ingest/pipeline

SUBJECTS = [
    "compensation-benefits",     # pay, 401k, equity, perks
    "leave-and-time-off",        # PTO, parental leave, FMLA
    "severance-termination",     # severance formula, involuntary separation
    "career-and-titles",         # leveling, titles, promotion criteria (IC side)
    "onboarding",                # new-hire ramp, both employee and manager-facing
    "workplace-conduct-policy",  # moonlighting, devices/security, code of conduct
    "internal-systems-tools",    # Okta, Jira, Slack, GitHub, etc.
    "team-rituals-meetings",     # how-we-work, recurring meetings, remote norms
    "manager-1-1s",              # 1:1 cadence and agenda
    "hiring",                    # recruiting, interviewing, offers
    "coaching-feedback",         # giving feedback, coaching skills
    "performance-management",    # reviews, performance model, underperformance process
    "promotion-recognition",     # promotion process, recognition programs
    "difficult-conversations",   # boundaries, hard conversations, terminations logistics
]

_TAG_TOOL = {
    "toolSpec": {
        "name": "submit_subjects",
        "description": "Submit the subject labels that apply.",
        "inputSchema": {"json": {
            "type": "object", "additionalProperties": False, "required": ["subjects"],
            "properties": {
                "subjects": {
                    "type": "array",
                    "items": {"type": "string", "enum": SUBJECTS},
                    "description": "1-3 subjects from the fixed vocabulary that best apply.",
                },
            },
        }},
    }
}


def _model_id() -> str:
    import os
    return os.environ["BEDROCK_MODEL_ID"]


def tag_subjects(text: str) -> list[str]:
    """Tag a chunk of text (or a query) with 1-3 subjects from the fixed vocabulary."""
    resp = bedrock.converse(
        modelId=_model_id(),
        system=[{"text": (
            "Classify the TEXT with 1 to 3 subjects from the fixed vocabulary given by "
            "the tool schema. Pick only subjects that are clearly on-topic."
        )}],
        messages=[{"role": "user", "content": [{"text": text}]}],
        toolConfig={"tools": [_TAG_TOOL], "toolChoice": {"tool": {"name": "submit_subjects"}}},
        inferenceConfig={"temperature": 0.0},
    )
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            subjects = block["toolUse"]["input"].get("subjects", [])
            return [s for s in subjects if s in SUBJECTS]
    return []
