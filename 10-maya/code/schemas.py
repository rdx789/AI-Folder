"""Project concepts, independent of model providers and transport libraries."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class CallerContext:
    """Supplied by authenticated application runtime, never by the model."""

    employee_id: str
    user_group: str


@dataclass(frozen=True)
class EvidenceCitation:
    source: str
    chunk_id: str
    section: str | None = None
    source_id: str | None = None
    collection: str | None = None
    audience: tuple[str, ...] = ()
    subject_employee_id: str | None = None
    sensitivity: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class Evidence:
    text: str
    citation: EvidenceCitation


@dataclass
class ContextPlan:
    current_intent: str
    subject_employee_id: str = "E001"
    relevant_facts: list[str] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    retrieval_query: str | None = None
    requires_retrieval: bool = False
    requires_operational_reads: bool = False
    also_needs: list[str] = field(default_factory=list)


@dataclass
class ConversationMemory:
    current_intent: str = ""
    important_facts: dict[str, str] = field(default_factory=dict)
    active_constraints: dict[str, str] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    closed_topics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChecklistTask:
    description: str
    status: Literal["proposed", "complete", "blocked", "pending", "unresolved"]
    evidence: tuple[EvidenceCitation, ...] = ()
    task_id: str = ""
    support_quotes: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status in {"proposed", "blocked"} and not self.evidence:
            raise ValueError("Proposed and blocked tasks require evidence")


@dataclass
class OnboardingChecklist:
    employee_id: str
    role: str | None = None
    start_date: date | None = None
    tasks: list[ChecklistTask] = field(default_factory=list)
    full_name: str = "Maya Cohen"
    location: str | None = None
    work_mode: str | None = None
    active_constraints: list[str] = field(default_factory=list)
    evidence: list[EvidenceCitation] = field(default_factory=list)
    status: Literal["ok", "unresolved", "denied", "pending"] = "ok"
    message: str = ""
    handoff: "WebexHandoff | None" = None
    pending_access: "PendingAccessResult | None" = None
    related_status: list[ChecklistTask] = field(default_factory=list)

    @property
    def blocked_items(self) -> list[ChecklistTask]:
        return [t for t in self.tasks if t.status == "blocked"]


@dataclass(frozen=True)
class WebexHandoff:
    caller: CallerContext
    employee_id: str
    business_reason: str
    idempotency_key: str
    system: Literal["Webex"] = "Webex"
    evidence: tuple[EvidenceCitation, ...] = ()
    kind: Literal["webex_access"] = "webex_access"


@dataclass(frozen=True)
class ReadToolCall:
    name: str
    arguments: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelTurn:
    tool_calls: list[ReadToolCall] = field(default_factory=list)
    tasks: list[ChecklistTask] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class PendingAccessResult:
    request_id: str
    status: Literal["pending"] = "pending"

    def __post_init__(self) -> None:
        if self.status != "pending" or not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("Access result must contain a request ID and pending status")


@dataclass
class WebexDispatchRecord:
    """Thread-scoped attempt and acknowledgement, outside model-owned memory."""

    handoff: WebexHandoff
    status: Literal["dispatching", "pending", "unresolved"] = "dispatching"
    result: PendingAccessResult | None = None
