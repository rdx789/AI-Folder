"""Top-level router: classify a NovaOps request, route it to exactly one handler.

Clean single-path routing — a `classify` node labels the request, a conditional edge
sends it to ONE handler. No Send, no reducers, no parallel dispatch. Each handler is
a focused `create_agent` over just the tools that category needs, or a plain node.

    python router.py                 # route three sample requests to three handlers
    python router.py "<request>"     # route and answer one request
"""

import asyncio
import sys
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import get_llm
from mcp_client import discover_tools

CategoryName = Literal["policy_qa", "it_ticket", "access_request", "lookup", "general"]


def trace(step: str, detail: str = "") -> None:
    print(f"[trace] {step}{' -> ' + detail if detail else ''}")


class Category(BaseModel):
    category: CategoryName
    reasoning: str = Field(default="", description="One line: why this category.")


class RouterState(TypedDict, total=False):
    request: str
    category: str
    handler: str
    answer: str


# Each agent handler: the discovered tools it may use, and how it should behave.
_HANDLERS = {
    "policy_qa": {
        "tools": ["list_policies", "get_policy"],
        "prompt": (
            "Answer the question using only the NovaOps policy documents. Name the "
            "policy you used. 2-4 sentences."
        ),
    },
    "it_ticket": {
        "tools": ["search_knowledge_base"],
        "prompt": (
            "Answer this IT how-to / troubleshooting question from the knowledge "
            "base. 2-4 sentences."
        ),
    },
    "lookup": {
        "tools": ["get_employee"],
        "prompt": "Look up the employee and say who they are in 1-2 sentences.",
    },
    "access_request": {
        "tools": [
            "get_employee",
            "check_software_subscription",
            "list_policies",
            "get_policy",
            "create_access_request",
        ],
        "prompt": (
            "Help resolve an access request. Identify the employee, check the "
            "software's seat availability and the access policy, and — only if the "
            "employee is clearly identified and eligible — file the access request. "
            "If you cannot identify the employee, ask for their employee id instead. "
            "Finish with a 2-4 sentence summary of what you did."
        ),
    },
}

_CLASSIFY_PROMPT = (
    "Classify this NovaOps request into exactly one category:\n"
    "- policy_qa: a question about what a company policy says\n"
    "- it_ticket: an IT how-to / troubleshooting question (login, VPN, MFA, laptop)\n"
    "- access_request: someone needs access to a system, tool, or license\n"
    "- lookup: look up an employee record\n"
    "- general: anything else, or unclear\n\n"
    "Request: {request}"
)


def build_router(tools_by_name: dict):
    llm = get_llm()

    agents = {}
    for name, spec in _HANDLERS.items():
        selected = [tools_by_name[t] for t in spec["tools"] if t in tools_by_name]
        agents[name] = create_agent(llm, selected, system_prompt=spec["prompt"])

    def classify(state: RouterState) -> dict:
        try:
            result = llm.with_structured_output(Category).invoke(
                _CLASSIFY_PROMPT.format(request=state["request"])
            )
            category, why = result.category, result.reasoning
        except Exception as exc:
            trace("classify", f"failed ({exc}); labelling 'general'")
            category, why = "general", "classification failed"
        trace("classify", f"category={category}" + (f" ({why})" if why else ""))
        return {"category": category}

    def route(state: RouterState) -> str:
        return state["category"]

    async def _run_agent(state: RouterState, name: str) -> dict:
        trace(name, "handling")
        try:
            result = await agents[name].ainvoke(
                {"messages": [HumanMessage(content=state["request"])]}
            )
            answer = result["messages"][-1].content
        except Exception as exc:
            answer = f"(handler '{name}' failed: {exc})"
        return {"handler": name, "answer": answer}

    async def policy_qa_handler(state: RouterState) -> dict:
        return await _run_agent(state, "policy_qa")

    async def it_ticket_handler(state: RouterState) -> dict:
        return await _run_agent(state, "it_ticket")

    async def lookup_handler(state: RouterState) -> dict:
        return await _run_agent(state, "lookup")

    async def access_request_handler(state: RouterState) -> dict:
        return await _run_agent(state, "access_request")

    def general_handler(state: RouterState) -> dict:
        trace("general", "handling (deterministic)")
        return {
            "handler": "general",
            "answer": (
                "I can help with NovaOps policy questions, IT how-to / troubleshooting, "
                "access requests, and employee lookups. Could you rephrase your request "
                "to point at one of those?"
            ),
        }

    g = StateGraph(RouterState)
    g.add_node("classify", classify)
    g.add_node("policy_qa", policy_qa_handler)
    g.add_node("it_ticket", it_ticket_handler)
    g.add_node("lookup", lookup_handler)
    g.add_node("access_request", access_request_handler)
    g.add_node("general", general_handler)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        route,
        ["policy_qa", "it_ticket", "lookup", "access_request", "general"],
    )
    for handler in _HANDLERS:
        g.add_edge(handler, END)
    g.add_edge("general", END)

    return g.compile()


DEMO_REQUESTS = [
    "what's the expense policy?",
    "I need Webex access",
    "who is E010?",
]


async def _run(requests: list[str]) -> None:
    tools = await discover_tools()
    print(f"Discovered {len(tools)} MCP tools.\n")
    router = build_router(tools)
    for request in requests:
        print(f"You: {request}")
        result = await router.ainvoke({"request": request})
        print(f"[router] handler = {result.get('handler')}")
        print(f"Assistant: {result.get('answer', '(no answer produced)')}\n")


def main() -> None:
    requests = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_REQUESTS
    try:
        asyncio.run(_run(requests))
    except RuntimeError as exc:  # e.g. MCP server unreachable
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
