"""Thread-scoped state; raw history is retained, model context is projected."""

from dataclasses import dataclass, field
from typing import Literal

from .schemas import (CallerContext, ContextPlan, ConversationMemory, Evidence, ChecklistTask,
                      OnboardingChecklist, ModelTurn, WebexDispatchRecord, WebexHandoff)


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: tuple[str, ...] = ()


@dataclass
class MayaState:
    thread_id: str
    caller: CallerContext
    messages: list[Message] = field(default_factory=list)
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    distilled_upto: int = 0
    plan: ContextPlan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    selected_tools: tuple[str, ...] = ()
    rearmed: bool = False
    retrieval_refinements: int = 0
    distill_hold: int = 0
    tool_rounds: int = 0
    reads_executed: int = 0
    draft: ModelTurn | None = None
    checklist: OnboardingChecklist | None = None
    missing_evidence: tuple[str, ...] = ()
    errors: list[str] = field(default_factory=list)
    handoff_records: dict[str, WebexDispatchRecord] = field(default_factory=dict)
    handoff_events: list[WebexHandoff] = field(default_factory=list)
    dispatch_key: str | None = None
    node_visits: list[str] = field(default_factory=list)
    known_tasks: list[ChecklistTask] = field(default_factory=list)
    unresolved_tasks: list[ChecklistTask] = field(default_factory=list)
    trivial_turn: bool = False
