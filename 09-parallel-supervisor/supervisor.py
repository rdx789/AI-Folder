"""The LangGraph supervisor: plan -> workers -> review (reflect) -> write gate -> finalize."""

import operator
import re
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, Field, ValidationError

from config import get_llm
from specialists import build_specialists

SpecialistName = Literal["knowledge", "records", "eligibility"]

MAX_ROUNDS = 2


def trace(step: str, detail: str = "") -> None:
    """One line per node as it runs — the real path to the answer, in order."""
    print(f"[trace] {step}{' -> ' + detail if detail else ''}")


# --- structured-output schemas --------------------------------------------------

class Plan(BaseModel):
    specialists: list[SpecialistName] = Field(
        default_factory=list,
        description="The specialists needed for this request. Pick only what is required.",
    )
    reasoning: str = Field(
        default="", description="Brief explanation of why these specialists were chosen."
    )


class Review(BaseModel):
    verdict: Literal["enough", "need_more"]
    follow_up_request: str = Field(
        default="",
        description="If need_more: a sharper, more specific version of the request for a second pass.",
    )


class WriteProposal(BaseModel):
    needs_write: bool = Field(description="True only if an access request must be filed.")
    employee_id: str = ""
    software: str = ""
    business_justification: str = ""
    reasoning: str = Field(
        default="", description="Why this write is (or is not) needed, from the findings."
    )


# --- state --------------------------------------------------------------------

class SupervisorState(TypedDict, total=False):
    request: str
    plan: list[str]
    round: int
    follow_up_request: str
    specialist_outputs: Annotated[list[dict], operator.add]
    write_proposal: dict
    approved: bool
    approval_reason: Optional[str]
    answer: str


# --- graph build -------------------------------------------------------------

def build_graph(tools_by_name: dict, checkpointer):
    """Compile the supervisor graph against a caller-supplied checkpointer.

    The checkpointer is passed in (not created here) so the durable one — an
    AsyncSqliteSaver held open by an `async with` in the caller — stays alive for
    the whole run.
    """
    llm = get_llm()
    specialists = build_specialists(tools_by_name)
    available = set(specialists)
    create_access_request = tools_by_name.get("create_access_request")

    def plan_node(state: SupervisorState) -> dict:
        prompt = (
            "You route a NovaOps request to a subset of these specialists:\n"
            "- knowledge: IT how-to / troubleshooting articles (password, VPN, MFA, laptops)\n"
            "- records: employee records and SaaS seat / subscription usage\n"
            "- eligibility: reads the company policy documents and judges access "
            "eligibility — pick this for ANY question about what a NovaOps policy says\n\n"
            f"Request: {state['request']}\n\n"
            "Return the specialists required — no more than needed."
        )
        reasoning = ""
        try:
            plan = llm.with_structured_output(Plan).invoke(prompt)
            chosen = [s for s in plan.specialists if s in available]
            reasoning = plan.reasoning
        except Exception as exc:
            trace("plan", f"structured output failed ({exc}); falling back to knowledge")
            chosen = []
        if not chosen:
            chosen = ["knowledge"] if "knowledge" in available else sorted(available)[:1]
        trace("plan", f"specialists={chosen}" + (f" ({reasoning})" if reasoning else ""))
        return {"plan": chosen, "round": 1}

    def dispatch(state: SupervisorState):
        ask = state.get("follow_up_request") or state["request"]
        rnd = state.get("round", 1)
        names = state["plan"]
        trace("dispatch", f"sending to {len(names)} specialist(s) in parallel: {names}")
        return [
            Send("worker", {"specialist": name, "ask": ask, "round": rnd})
            for name in names
        ]

    async def worker(payload: dict) -> dict:
        name = payload["specialist"]
        agent = specialists[name]
        trace("worker", f"'{name}' running...")
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=payload["ask"])]}
            )
            summary = result["messages"][-1].content
        except Exception as exc:
            summary = f"(specialist '{name}' failed: {exc})"
        trace("worker", f"'{name}' done ({len(summary)} chars)")
        return {
            "specialist_outputs": [
                {"agent": name, "summary": summary, "round": payload["round"]}
            ]
        }

    def _extract_request_id(result) -> str:
        """Pull a request_id out of the MCP tool's return, whatever shape it takes."""
        text = result if isinstance(result, str) else str(result)
        m = re.search(r'"request_id"\s*:\s*"([^"]+)"', text)
        return m.group(1) if m else ""

    def _current_outputs(state: SupervisorState) -> list[dict]:
        outs = state.get("specialist_outputs", [])
        if not outs:
            return []
        latest = max(o["round"] for o in outs)
        return [o for o in outs if o["round"] == latest]

    def _findings_text(state: SupervisorState) -> str:
        return "\n".join(
            f"- {o['agent']}: {o['summary']}" for o in _current_outputs(state)
        )

    def review_node(state: SupervisorState) -> dict:
        prompt = (
            f"Original request: {state['request']}\n\n"
            f"Specialist findings so far:\n{_findings_text(state)}\n\n"
            "Can the request be answered from these findings? Default to 'enough'. "
            "Answer 'need_more' ONLY if a specialist explicitly could not retrieve a "
            "concrete fact that is required to act (e.g. a missing employee id, an "
            "unresolved seat count, a policy that wasn't found). Do not ask for more "
            "just to add polish or detail. "
            "If 'need_more', write a sharper follow-up naming exactly the missing fact."
        )
        try:
            review = llm.with_structured_output(Review).invoke(prompt)
        except Exception as exc:
            trace("review", f"structured output failed ({exc}); treating as enough")
            return {}
        rnd = state.get("round", 1)
        if review.verdict == "need_more" and rnd < MAX_ROUNDS:
            follow_up = review.follow_up_request or state["request"]
            trace("review", f"need_more (round {rnd}): {follow_up}")
            return {"round": rnd + 1, "follow_up_request": follow_up}
        trace("review", f"enough (round {rnd})")
        return {"round": rnd}  # unchanged; marks the decision as 'enough'

    def route_review(state: SupervisorState):
        outs = state.get("specialist_outputs", [])
        answered_round = max((o["round"] for o in outs), default=0)
        # round was bumped past the round the specialists last answered -> go again
        if state.get("round", 1) > answered_round:
            ask = state["follow_up_request"]
            rnd = state["round"]
            names = state["plan"]
            trace("dispatch", f"re-dispatching to {len(names)} specialist(s): {names}")
            return [
                Send("worker", {"specialist": name, "ask": ask, "round": rnd})
                for name in names
            ]
        return "propose_write"

    def propose_write_node(state: SupervisorState) -> dict:
        prompt = (
            "Based ONLY on the specialists' findings below, decide whether resolving this "
            "request requires filing a NovaOps access request (a WRITE).\n\n"
            f"Request: {state['request']}\n\n"
            f"Findings:\n{_findings_text(state)}\n\n"
            "Set needs_write=true ONLY if the findings name a specific, verified employee "
            "id (like 'E010') and a single specific software. If the employee or software "
            "is unknown or ambiguous, set needs_write=false. Never use a placeholder. "
            "When true, give employee_id, software, a one-line business_justification, "
            "and your reasoning."
        )
        try:
            proposal = llm.with_structured_output(WriteProposal).invoke(prompt)
        except (ValidationError, Exception) as exc:
            trace("propose_write", f"proposal failed ({exc}); treating as no write")
            return {"write_proposal": {"needs_write": False}}

        emp_ok = bool(re.fullmatch(r"[A-Za-z0-9-]{2,20}", proposal.employee_id or ""))
        sw_ok = bool(proposal.software) and "/" not in proposal.software
        placeholder = re.search(r"[A-Z]{3,}_[A-Z]{3,}|required|placeholder|unknown|tbd",
                                f"{proposal.employee_id} {proposal.software}", re.I)
        # The software must be one the user actually named — the model sometimes
        # invents a write for a plain lookup.
        asked = f"{state['request']} {state.get('follow_up_request', '')}".lower()
        named = bool(proposal.software) and proposal.software.lower() in asked
        if proposal.needs_write and (
            not emp_ok or not sw_ok or placeholder or not named
            or not proposal.business_justification
        ):
            trace("propose_write", "proposal not concrete enough; downgraded to no write")
            return {"write_proposal": {"needs_write": False}}

        if proposal.needs_write:
            trace(
                "propose_write",
                f"WRITE needed: create_access_request("
                f"employee_id='{proposal.employee_id}', software='{proposal.software}')",
            )
        else:
            trace("propose_write", "no write needed")
        return {"write_proposal": proposal.model_dump()}

    def route_write(state: SupervisorState):
        return "approval_gate" if state["write_proposal"].get("needs_write") else "finalize"

    def approval_gate_node(state: SupervisorState) -> dict:
        p = state["write_proposal"]
        decision = interrupt(
            {
                "action": "create_access_request",
                "employee_id": p["employee_id"],
                "software": p["software"],
                "business_justification": p["business_justification"],
                "reasoning": p.get("reasoning", ""),
            }
        )
        approved = bool(decision.get("approved"))
        reason = decision.get("reason")
        if not approved and not (reason and reason.strip()):
            raise ValueError("A denied write must include a reason.")
        trace("approval_gate", "APPROVED" if approved else f"DENIED: {reason}")
        return {"approved": approved, "approval_reason": reason}

    def route_approval(state: SupervisorState):
        return "execute_write" if state.get("approved") else "finalize"

    async def execute_write_node(state: SupervisorState) -> dict:
        p = state["write_proposal"]
        trace("execute_write", f"calling create_access_request for {p['employee_id']}...")
        try:
            result = await create_access_request.ainvoke(
                {
                    "employee_id": p["employee_id"],
                    "software": p["software"],
                    "business_justification": p["business_justification"],
                }
            )
            trace("execute_write", f"done: {result}")
            return {"write_proposal": {**p, "result": result, "filed": True}}
        except Exception as exc:
            trace("execute_write", f"failed: {exc}")
            return {"write_proposal": {**p, "error": str(exc), "filed": False}}

    def finalize_node(state: SupervisorState) -> dict:
        p = state.get("write_proposal", {})
        write_note = ""
        status_line = ""
        if p.get("filed"):
            rid = _extract_request_id(p.get("result"))
            ref = f" (reference {rid})" if rid else ""
            status_line = (
                f"An access request for {p['software']} has ALREADY been filed for "
                f"{p['employee_id']}{ref} and is pending approval. The user needs to take "
                "no further action — do not tell them to contact IT or open a ticket."
            )
            write_note = f"\n\nAccess request filed{ref} — status: pending approval."
        elif p.get("needs_write") and not p.get("filed"):
            status_line = (
                "An access request was proposed but NOT filed (a human denied it: "
                f"{state.get('approval_reason')}). Explain that access was not granted."
            )
            write_note = (
                f"\n\nNo access request was filed (reason: {state.get('approval_reason')})."
            )
        prompt = (
            f"Request: {state['request']}\n\n"
            f"Specialist findings:\n{_findings_text(state)}\n\n"
            f"{status_line}\n\n"
            "Write one clear, direct answer to the request."
        )
        trace("finalize", "merging the specialists' findings into final answer")
        try:
            answer = llm.invoke(prompt).content
        except Exception as exc:
            answer = f"(could not compose final answer: {exc})\n\n{_findings_text(state)}"
        return {"answer": answer + write_note}

    g = StateGraph(SupervisorState)
    g.add_node("plan", plan_node)
    g.add_node("worker", worker)
    g.add_node("review", review_node)
    g.add_node("propose_write", propose_write_node)
    g.add_node("approval_gate", approval_gate_node)
    g.add_node("execute_write", execute_write_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", dispatch, ["worker"])
    g.add_edge("worker", "review")
    g.add_conditional_edges("review", route_review, ["worker", "propose_write"])
    g.add_conditional_edges("propose_write", route_write, ["approval_gate", "finalize"])
    g.add_conditional_edges("approval_gate", route_approval, ["execute_write", "finalize"])
    g.add_edge("execute_write", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
