"""Three specialist sub-agents, each bound only to its slice of the discovered tools."""

from langchain.agents import create_agent

from config import get_llm

# Which discovered tool each specialist is allowed to touch. Names come from the
# server at runtime; this only says who gets which.
SPECIALIST_TOOLS = {
    "knowledge": ["search_knowledge_base"],
    "records": ["get_employee", "check_software_subscription"],
    "eligibility": ["get_employee", "list_policies", "get_policy"],
}

SPECIALIST_ROLE = {
    "knowledge": "You answer IT how-to and troubleshooting questions from the NovaOps knowledge base.",
    "records": "You look up employee records and SaaS seat/subscription usage.",
    "eligibility": "You judge whether an employee is eligible for access, citing NovaOps policy.",
}

# Shared work-bounding clause appended to every specialist prompt.
_BOUND = (
    " Make at most three tool calls, then stop and give a 2-3 sentence answer "
    "based on what you found."
)


def _prompt(name: str) -> str:
    return SPECIALIST_ROLE[name] + _BOUND


def build_specialists(tools_by_name: dict) -> dict:
    """Return {specialist_name: compiled agent}, each wired to its tool subset."""
    llm = get_llm()
    agents = {}
    for name, wanted in SPECIALIST_TOOLS.items():
        selected = [tools_by_name[t] for t in wanted if t in tools_by_name]
        if not selected:
            # The server didn't expose anything this specialist can use — skip it
            # rather than register a toolless agent.
            continue
        agents[name] = create_agent(llm, selected, system_prompt=_prompt(name))
    return agents
