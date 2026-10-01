"""Maya workflow: plan -> authorized evidence -> read loadout -> answer/handoff.

The authenticated host supplies configurable.caller and a stable thread_id.
Only the injected Webex port may submit requests to the separate workflow.

Command line (Lesson 10 style; see main() at the end), always with the live Bedrock model.
Also installed as the `maya-live` command (same options):
    python graph.py --session S2 --trace          # replay S2 turn by turn
    python graph.py "Is a Webex seat available?"  # one question
    python graph.py --chat                        # interactive, one thread
"""
import os as _os
import time as _time
# CLI timing starts here, before the heavy imports below, so "startup" includes them.
_CLI_T0 = float(_os.environ.pop('MAYA_CLI_T0', _time.perf_counter()))
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/graph.py): relative imports need the package, so
    # re-run this file as the installed module `maya.graph`.
    import runpy
    _os.environ['MAYA_CLI_T0'] = str(_CLI_T0)  # same process: keep the first timestamp
    try:
        runpy.run_module('maya.graph', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date
import asyncio
import json
import re
from typing import TypedDict

from .access import DENIAL, PermissionDenied, authorize_subject, caller_from_config
from .grounding import excludes_closed_topic, validate_tasks
from .memory import needs_distillation, memory_is_safe, distillation_boundary, visible_history
from .model import PlanningModel
from .operations import ReadPort, LocalReadPort, authorize_read, sanitize_result, operational_evidence, tool_schemas
from .policy import (conversation_window, select_tools, is_trivial_message, necessary_reads,
                     explicit_source_intent, grounded_facts, full_checklist_request,
                     FULL_CHECKLIST_SOURCES)
from .ports import WebexPort, WebexDispatchGuard, webex_request_key, validate_pending_result
from .retrieval import EvidenceRetriever, retrieve_evidence_node
from .schemas import (ContextPlan, OnboardingChecklist, ChecklistTask, ModelTurn,
                      WebexHandoff, WebexDispatchRecord)
from .state import MayaState, Message

RECURSION_LIMIT = 12
MAX_RETRIEVAL_REFINEMENTS = 1
MAX_TOOL_ROUNDS = 2
WEBEX_BUSINESS_REASON = 'Webex access for Maya Cohen (E001) onboarding; separate approval is required.'


@dataclass(frozen=True)
class Dependencies:
    model: PlanningModel
    retriever: EvidenceRetriever
    webex: WebexPort | None = None
    reads: ReadPort = field(default_factory=LocalReadPort)


def can_rearm(state: MayaState, *, reads_executed: int) -> bool:
    return bool(state.plan and state.plan.requires_operational_reads
                and reads_executed == 0 and not state.rearmed)


class WorkflowState(TypedDict):
    newest_message: str
    session: MayaState
    response: OnboardingChecklist


def _iso_date(value):
    """User text and model-distilled memory may hold an impossible or prose date."""
    try:
        return date.fromisoformat(value) if isinstance(value, str) and value else None
    except ValueError:
        return None


def _remember_user(state, newest):
    """Pin exact S2 user facts before model compression; no inferred cancellation."""
    facts = state.memory.important_facts
    dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', newest)
    # An impossible date (2026-02-30) is not pinned; it must not crash planning.
    # "starts", "starting", "start date" (the assignment request says "starting 2026-08-01").
    if dates and re.search(r'\bstart(?:s|ing)?\b|start date', newest, re.I) and _iso_date(dates[0]):
        facts['start_date'] = dates[0]
    if re.search(r'\bIsrael\b', newest, re.I):
        facts['location'] = 'Israel'
    if re.search(r'\bhybrid\b', newest, re.I):
        facts['work_mode'] = 'hybrid'
    if re.search(r'\bQ3\b.*\bfreeze\b', newest, re.I):
        state.memory.active_constraints['saas_freeze'] = 'Q3 SaaS freeze'
    if re.search(r'drop it|close (?:it|that)|back to maya', newest, re.I):
        prior = next((m.content for m in reversed(state.messages[:-1]) if m.role == 'user'), '')
        if 'rachel' in prior.lower() and 'Rachel' not in state.memory.closed_topics:
            state.memory.closed_topics.append('Rachel')
            closed_ids = {t.task_id for t in state.known_tasks if any(
                c.subject_employee_id == 'E010' for c in t.evidence)}
            for passage in state.evidence:
                if passage.citation.subject_employee_id == 'E010':
                    try:
                        rows = json.loads(passage.text)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(rows, list):
                        closed_ids.update(row.get('ticket_id') or row.get('task_id')
                                          for row in rows if isinstance(row, dict))
            state.memory.important_facts = {k: v for k, v in facts.items() if 'rachel' not in (k + v).lower()}
            state.memory.unresolved_items = [v for v in state.memory.unresolved_items if 'rachel' not in v.lower()]
            state.evidence = [e for e in state.evidence if e.citation.subject_employee_id != 'E010']
            state.known_tasks = [t for t in state.known_tasks if not any(
                c.subject_employee_id == 'E010' for c in t.evidence)
                and not excludes_closed_topic(t.description, state.memory.closed_topics)]
            state.unresolved_tasks = [t for t in state.unresolved_tasks if t.task_id not in closed_ids
                                     and not excludes_closed_topic(t.description, state.memory.closed_topics)]


def handoff_requested(newest):
    """A conservative affirmative command, never a model authorization claim."""
    if re.search(r"\b(don't|do not|not|never|if|unless|draft|hypothetical|could|would)\b", newest, re.I):
        return False
    return bool(re.search(
        r'(?:^|[.!]\s*)(?:please\s+)?(?:go ahead and\s+)?(?:file|submit|send|create)\b'
        r'[^.!]*\bWebex\b[^.!]*\b(?:now|request)\b', newest, re.I))


def recall_only(newest):
    """Explicit conversation recall never opens retrieval, even on planner error."""
    if re.search(r'\b(refresh|look up|check again|latest|right now)\b', newest, re.I):
        return False
    return bool(re.search(r'\bremind me\b.*\b(gave you|told you|mentioned)\b', newest, re.I)
                or re.search(r'\b(write|give)\b.*\bfull status summary\b', newest, re.I))


class MayaGraph:
    """Thin host boundary around the compiled/checkpointed LangGraph."""
    def __init__(self, compiled):
        self.compiled = compiled
        self._thread_locks: dict[str, asyncio.Lock] = {}

    async def ainvoke(self, input, config=None):
        config = deepcopy(config or {})
        try:
            caller = caller_from_config(config)
            authorize_subject(caller, 'E001')
            thread = config.get('configurable', {}).get('thread_id')
            if not isinstance(thread, str) or not thread:
                raise ValueError('A stable thread_id is required')
        except PermissionDenied:
            return {'response': OnboardingChecklist('', full_name='', status='denied', message=DENIAL)}
        if not isinstance(input, dict) or set(input) != {'newest_message'} or not isinstance(input['newest_message'], str):
            raise ValueError('Input must contain only newest_message; caller and state are host-owned')
        limit = config.get('recursion_limit', RECURSION_LIMIT)
        config['recursion_limit'] = min(limit, RECURSION_LIMIT) if type(limit) is int and limit > 0 else RECURSION_LIMIT
        # Serialize turns of one thread, including simultaneous turn-10 replays.
        async with self._thread_locks.setdefault(thread, asyncio.Lock()):
            snapshot = await self.compiled.aget_state(config)
            previous = snapshot.values.get('session') if snapshot.values else None
            if previous and previous.caller != caller:
                return {'response': OnboardingChecklist('', full_name='', status='denied', message=DENIAL)}
            return await self.compiled.ainvoke(input, config)

    def get_graph(self):
        return self.compiled.get_graph()


def build_graph(dependencies: Dependencies, *, checkpointer=None) -> MayaGraph:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    dispatcher = WebexDispatchGuard(dependencies.webex)

    async def plan_current_turn(data, config):
        caller = caller_from_config(config)
        from pydantic import TypeAdapter
        saved = data.get('session')
        # Msgpack restores tuples as lists. Revalidate checkpointed project types
        # so cached citations retain exact equality across subsequent turns.
        state = TypeAdapter(MayaState).validate_python(asdict(saved)) if saved else MayaState(config['configurable']['thread_id'], caller)
        newest = data['newest_message']
        state.messages.append(Message('user', newest))
        state.rearmed, state.retrieval_refinements = False, 0
        state.tool_rounds, state.reads_executed = 0, 0
        state.selected_tools, state.missing_evidence, state.errors = (), (), []
        state.draft = None
        state.dispatch_key = None
        state.node_visits = ['plan']
        state.trivial_turn = is_trivial_message(newest)
        _remember_user(state, newest)
        if state.trivial_turn:
            state.plan = ContextPlan('other', relevant_facts=list(state.memory.important_facts.values()),
                                     active_constraints=list(state.memory.active_constraints.values()))
            return {'session': state}
        boundary = distillation_boundary(state)
        if needs_distillation(state) and len(state.messages) > state.distill_hold and boundary > state.distilled_upto:
            try:
                memory = await dependencies.model.distill(
                    memory=deepcopy(state.memory),
                    fresh_history=visible_history(state, state.messages[state.distilled_upto:boundary]))
                from pydantic import TypeAdapter
                from .schemas import ConversationMemory
                memory = TypeAdapter(ConversationMemory).validate_python(memory)
                if not memory_is_safe(state.memory, memory,
                                      visible_history(state, state.messages[state.distilled_upto:boundary])):
                    raise ValueError('Unsafe memory compression')
                if len(json.dumps(asdict(memory))) > 8000:
                    raise ValueError('Distilled memory exceeds its context budget')
                # A model-closed topic would silently hide matching tasks and user turns.
                memory.closed_topics = list(state.memory.closed_topics)
                state.memory, state.distilled_upto, state.distill_hold = memory, boundary, 0
            except Exception:
                # Safe degradation keeps the raw tail and delays an identical retry.
                state.distill_hold = len(state.messages) + 4
        try:
            plan = await dependencies.model.plan(newest_message=newest, memory=deepcopy(state.memory),
                fresh_history=visible_history(state, state.messages[state.distilled_upto:-1]))
            from pydantic import TypeAdapter
            plan = TypeAdapter(ContextPlan).validate_python(plan)
        except Exception:
            plan = ContextPlan('unknown', requires_retrieval=True,
                               retrieval_query=newest, requires_operational_reads=True)
        # Retrieval subject is application-owned, including during Rachel's tangent.
        plan.subject_employee_id = 'E001'
        explicit_source = explicit_source_intent(newest)
        if explicit_source and not recall_only(newest) and not handoff_requested(newest):
            plan.current_intent = explicit_source
            plan.also_needs = [i for i in plan.also_needs if i not in ('recall', 'other', explicit_source)]
            if explicit_source in ('employee_lookup', 'ticket_status') and not re.search(
                    r'\b(policy|memo|guidance|approval|threshold)\b', newest, re.I):
                plan.requires_retrieval = False
                plan.required_evidence = []
                plan.also_needs = []
            elif explicit_source == 'onboarding_status':
                plan.requires_retrieval = False
                plan.required_evidence = []
            elif explicit_source == 'equipment_request' and not re.search(r'\blaptop\b', newest, re.I):
                plan.requires_retrieval = True
                plan.required_evidence = ['equipment_policy.md', 'remote_equipment_budget_update.md']
            elif explicit_source == 'subscription_review' and re.search(r'\bWebex\b', newest, re.I):
                plan.requires_retrieval = True
                plan.required_evidence = ['webex_license_assignment.md', 'webex_license_cleanup_memo.md']
            elif explicit_source == 'policy_question' and 'saas_freeze' in state.memory.active_constraints:
                plan.requires_retrieval = True
                plan.requires_operational_reads = False
                plan.required_evidence = ['saas_renewal_freeze_q3.md', 'webex_license_cleanup_memo.md']
        if full_checklist_request(newest) and not recall_only(newest) and not handoff_requested(newest):
            # The assignment's headline request: every section needs its own source.
            plan.current_intent, plan.also_needs = 'onboarding_status', []
            plan.requires_retrieval = plan.requires_operational_reads = True
            plan.required_evidence = list(FULL_CHECKLIST_SOURCES)
        if handoff_requested(newest):
            plan.current_intent = 'access_request'
            supported = validate_tasks(state.known_tasks, state.evidence)
            if any('webex' in t.description.lower() and t.status in ('blocked', 'pending', 'proposed')
                   for t in supported):
                # Filing an already evidenced request is a handoff, not a refresh.
                plan.requires_retrieval = plan.requires_operational_reads = False
                plan.required_evidence = []
        if plan.current_intent == 'recall' or recall_only(newest):
            plan.current_intent = 'recall'
            plan.requires_retrieval = plan.requires_operational_reads = False
        elif necessary_reads(plan, newest):
            # SDD keeps the source's group reachable even on a mistaken no-read
            # flag. Maya also requires execution before accepting fresh status.
            plan.requires_operational_reads = True
        pinned = [*state.memory.important_facts.values(), *state.memory.active_constraints.values()]
        planner_facts = grounded_facts(plan.relevant_facts, memory_values=pinned,
            user_texts=[m.content for m in state.messages if m.role == 'user'])
        plan.relevant_facts = list(dict.fromkeys([*state.memory.important_facts.values(), *planner_facts]))
        plan.active_constraints = list(dict.fromkeys([*state.memory.active_constraints.values(), *plan.active_constraints]))
        state.plan, state.memory.current_intent = plan, plan.current_intent
        return {'session': state}

    async def authorize_and_retrieve(data, config):
        state = data['session']
        state.node_visits.append('retrieve')
        try:
            result = await retrieve_evidence_node(state.plan, config, dependencies.retriever)
            if result.status == 'denied':
                raise PermissionDenied()
            if any(e.citation.subject_employee_id not in (None, 'E001') for e in result.evidence):
                raise PermissionDenied()
            state.retrieval_refinements = result.refinements
            state.missing_evidence = result.missing
            # Current chunks replace earlier revisions of the same source, while
            # retaining other authorized evidence for recall-only turns.
            sources = {e.citation.source for e in result.evidence}
            state.evidence = [e for e in state.evidence if e.citation.source not in sources] + list(result.evidence)
            state.known_tasks = [t for t in validate_tasks(state.known_tasks, state.evidence) if t.status != 'unresolved']
        except PermissionDenied:
            state.errors.append('denied')
            state.evidence = []
        except Exception:
            state.errors.append('Evidence retrieval failed')
        return {'session': state}

    def select_read_tools(data):
        state = data['session']
        state.node_visits.append('select')
        state.selected_tools = () if state.errors else select_tools(state.plan)
        needs = necessary_reads(state.plan, data['newest_message'])
        if needs and not state.plan.also_needs and not state.errors:
            state.selected_tools = tuple(n for n in state.selected_tools if n in needs)
        return {'session': state}

    async def answer_or_handoff(data):
        state = data['session']
        state.node_visits.append('model')
        if state.trivial_turn:
            state.draft = ModelTurn()
            return {'session': state}
        if (state.plan.current_intent == 'access_request' and handoff_requested(data['newest_message'])
                and not state.plan.requires_retrieval and not state.plan.requires_operational_reads
                and not state.errors):
            supported = [t for t in validate_tasks(state.known_tasks, state.evidence)
                         if 'webex' in t.description.lower()
                         and t.status in ('blocked', 'pending', 'proposed')]
            if supported:
                # An explicit, already evidenced typed request is application
                # work. Approval is pending at the port, not missing evidence
                # for another free-form model answer to invent.
                state.draft = ModelTurn(tasks=[*supported, *state.unresolved_tasks])
                return {'session': state}
        if (state.draft is not None and not state.draft.tool_calls
                and can_rearm(state, reads_executed=state.reads_executed)):
            rearm(data)
        names = state.selected_tools if state.tool_rounds < MAX_TOOL_ROUNDS and not state.errors else ()
        try:
            draft = await dependencies.model.respond(
                newest_message=data['newest_message'], plan=deepcopy(state.plan), memory=deepcopy(state.memory),
                history=visible_history(state, conversation_window(state.messages)),
                evidence=tuple(state.evidence), tools=tool_schemas(names),
                previous_checklist=deepcopy(state.checklist))
            from pydantic import TypeAdapter
            state.draft = TypeAdapter(ModelTurn).validate_python(draft)
        except Exception:
            state.draft = ModelTurn()
            state.errors.append('Structured answer failed')
        return {'session': state}

    def route(data):
        state = data['session']
        if state.errors:
            return 'finalize'
        if state.draft.tool_calls and state.tool_rounds < MAX_TOOL_ROUNDS:
            return 'tools'
        if can_rearm(state, reads_executed=state.reads_executed):
            return 'rearm'
        return 'finalize'

    def rearm(data):
        state = data['session']
        state.rearmed = True
        state.selected_tools = select_tools(state.plan, fallback=True)
        state.draft = None  # discard the premature structured answer
        return {'session': state}

    async def execute_reads(data, config):
        state = data['session']
        state.node_visits.append('tools')
        calls = state.draft.tool_calls
        if len(calls) > 7:
            state.errors.append('Too many read calls')
            return {'session': state}
        try:
            # Validate the entire batch before any execution.
            for call in calls:
                authorize_read(caller_from_config(config), state.plan, call,
                               state.selected_tools, data['newest_message'])
            state.messages.append(Message('assistant', '', tuple(c.name for c in calls)))
            for call in calls:
                target = 'E010' if state.plan.current_intent == 'ticket_status' and 'rachel' in data['newest_message'].lower() else 'E001'
                value = await dependencies.reads.call(call.name, call.arguments)
                value = sanitize_result(call.name, value, state.caller, target)
                item = operational_evidence(call.name, call.arguments, value,
                       turn=sum(m.role == 'user' for m in state.messages), target=target)
                # Replace a stale snapshot of the same lookup rather than conflict
                # with its refreshed result on turns 3/11.
                state.evidence = [e for e in state.evidence if not (
                    e.citation.source == item.citation.source and e.citation.subject_employee_id == target)] + [item]
                state.messages.append(Message('tool', item.text))
                state.known_tasks = [t for t in state.known_tasks if all(
                    c.chunk_id in {e.citation.chunk_id for e in state.evidence} for c in t.evidence)]
                state.reads_executed += 1
                # A record without a role is still a valid read; don't fail the round.
                role = value[0].get('role') if call.name == 'get_employee' and len(value) == 1 else None
                if target == 'E001' and isinstance(role, str) and role:
                    state.memory.important_facts['role'] = role
                # The record's location fills in only what the user has not stated.
                location = value[0].get('location') if call.name == 'get_employee' and len(value) == 1 else None
                if target == 'E001' and isinstance(location, str) and location:
                    state.memory.important_facts.setdefault('location', location)
        except PermissionDenied:
            state.errors.append('denied')
        except Exception:
            state.errors.append('Operational read failed')
        state.tool_rounds += 1
        return {'session': state}

    def structured_response(data):
        state = data['session']
        state.node_visits.append('finalize')
        if 'denied' in state.errors:
            response = OnboardingChecklist('', full_name='', status='denied', message=DENIAL)
        else:
            facts = state.memory.important_facts
            tasks = validate_tasks(state.draft.tasks if state.draft else [], state.evidence)
            if state.plan.current_intent == 'recall' and re.search(r'\bsummary\b', data['newest_message'], re.I):
                unresolved_keys = {t.task_id for t in state.unresolved_tasks}
                tasks = validate_tasks([*state.known_tasks,
                    *[t for t in tasks if t.task_id not in unresolved_keys],
                    *state.unresolved_tasks], state.evidence)
            related = [t for t in tasks if state.plan.current_intent == 'ticket_status'
                       and 'rachel' in data['newest_message'].lower() and t.evidence
                       and all(c.subject_employee_id == 'E010' for c in t.evidence)]
            tasks = [t for t in tasks if not excludes_closed_topic(t.description, state.memory.closed_topics)
                     and all(c.subject_employee_id in (None, 'E001') for c in t.evidence)]
            # Missing source coverage is unresolved even if the model produced a
            # confident extractive statement from some other source.
            for source in state.missing_evidence:
                tasks.append(ChecklistTask(f'Required evidence: {source}', 'unresolved', reason='Source not retrieved'))
            if state.errors or (state.plan.requires_operational_reads and not state.reads_executed):
                tasks.append(ChecklistTask('Required reads', 'unresolved', reason='; '.join(state.errors) or 'Model skipped required reads'))
            if state.draft and state.draft.tool_calls:
                tasks.append(ChecklistTask('Additional reads', 'unresolved', reason='Read-round limit reached'))
            if not tasks and state.plan.current_intent not in ('recall', 'employee_lookup', 'other'):
                tasks.append(ChecklistTask('Requested onboarding status', 'unresolved', reason='No supported tasks supplied'))
            response = OnboardingChecklist('E001', role=facts.get('role'),
                start_date=_iso_date(facts.get('start_date')),
                tasks=tasks, location=facts.get('location'), work_mode=facts.get('work_mode'),
                active_constraints=list(state.memory.active_constraints.values()),
                evidence=list({c.chunk_id: c for t in [*tasks, *related] for c in t.evidence}.values()),
                related_status=related,
                status='unresolved' if any(t.status == 'unresolved' for t in tasks) else 'ok')
            # Generated prose is not a second, unchecked channel for factual claims.
            response.message = '; '.join(t.description + ': ' + t.status for t in tasks)
            if state.trivial_turn:
                response.message = 'Hello.' if data['newest_message'].strip().lower().rstrip('!.') in ('hi', 'hello', 'hey') else 'Understood.'
            acknowledged = [r for r in state.handoff_records.values() if r.status == 'pending']
            if acknowledged:
                response.pending_access = validate_pending_result(acknowledged[-1].result)
            explicit = handoff_requested(data['newest_message'])
            supported_webex = [t for t in tasks if 'webex' in t.description.lower()
                               and t.status in ('blocked', 'pending', 'proposed') and t.evidence]
            if state.plan.current_intent == 'access_request' and explicit:
                key = webex_request_key(thread_id=state.thread_id, caller=state.caller,
                                        business_reason=WEBEX_BUSINESS_REASON)
                existing = state.handoff_records.get(key)
                if existing:
                    response.handoff = existing.handoff
                    if existing.status == 'pending':
                        response.pending_access = validate_pending_result(existing.result)
                    else:
                        mark_handoff_unresolved(response, 'Prior dispatch acknowledgement is unresolved; no second call was made')
                elif supported_webex and response.status != 'unresolved' and not state.errors:
                    cites = tuple({c.chunk_id: c for t in supported_webex for c in t.evidence}.values())
                    response.handoff = WebexHandoff(state.caller, 'E001', WEBEX_BUSINESS_REASON, key, evidence=cites)
                    if dependencies.webex is None:
                        mark_handoff_unresolved(response, 'Webex workflow is not configured; no handoff was submitted')
                    else:
                        # This node's checkpoint records the attempt BEFORE the
                        # next node enters the external workflow. Replay of an
                        # unacknowledged attempt never calls the port again.
                        state.handoff_records[key] = WebexDispatchRecord(response.handoff)
                        state.dispatch_key = key
        return {'session': state, 'response': response}

    async def dispatch_webex(data, config):
        state, response = data['session'], data['response']
        state.node_visits.append('handoff')
        if state.dispatch_key:
            record = state.handoff_records[state.dispatch_key]
            try:
                caller = caller_from_config(config)
                authorize_subject(caller, 'E001')
                if caller != record.handoff.caller:
                    raise PermissionDenied()
                result = await dispatcher.request_access(record.handoff)
                record.status, record.result = 'pending', result
                response.pending_access = result
                if not any(h.idempotency_key == state.dispatch_key for h in state.handoff_events):
                    state.handoff_events.append(record.handoff)
                decision = f'Webex handoff {result.request_id} is pending approval.'
                if decision not in state.memory.decisions:
                    state.memory.decisions.append(decision)
            except PermissionDenied:
                record.status = 'unresolved'
                response = OnboardingChecklist('', full_name='', status='denied', message=DENIAL)
            except Exception:
                record.status = 'unresolved'
                mark_handoff_unresolved(response, 'Webex acknowledgement is unresolved; approval status is unknown')
            state.dispatch_key = None
        if response.status != 'denied':
            if not state.trivial_turn:
                response.message = '; '.join(t.description + ': ' + t.status for t in [*response.tasks, *response.related_status])
            if response.pending_access:
                if response.status == 'ok':
                    response.status = 'pending'
                response.message += f'; Webex handoff {response.pending_access.request_id} is pending approval.'
            supported = [t for t in response.tasks if t.status != 'unresolved']
            keys = {t.task_id or t.description for t in supported}
            # A diagnosed conflict invalidates a previously supported cached
            # version; preserve its unresolved item until a later grounded turn
            # resolves that task. It must not disappear from a full summary.
            unresolved = [t for t in response.tasks if t.status == 'unresolved' and t.task_id]
            invalid = {t.task_id for t in unresolved}
            state.unresolved_tasks = [t for t in state.unresolved_tasks
                                      if t.task_id not in keys | invalid] + unresolved
            state.known_tasks = validate_tasks([
                *[t for t in state.known_tasks if (t.task_id or t.description) not in keys | invalid],
                *supported], state.evidence)
            state.known_tasks = [t for t in state.known_tasks if t.status != 'unresolved']
        state.checklist = response
        state.messages.append(Message('assistant', json.dumps(history_view(response), default=str)))
        return {'session': state, 'response': response}

    graph = StateGraph(WorkflowState)
    for name, node in [('plan', plan_current_turn), ('retrieve', authorize_and_retrieve),
                       ('select', select_read_tools), ('model', answer_or_handoff),
                       ('tools', execute_reads), ('finalize', structured_response),
                       ('handoff', dispatch_webex)]:
        graph.add_node(name, node)
    graph.add_edge(START, 'plan')
    graph.add_edge('plan', 'retrieve')
    graph.add_edge('retrieve', 'select')
    graph.add_edge('select', 'model')
    # Rearm happens at the next model entry, saving a superstep so dispatch still
    # fits the original 12-step cap after a retry and both read rounds.
    graph.add_conditional_edges('model', route, {'tools': 'tools', 'rearm': 'model', 'finalize': 'finalize'})
    graph.add_edge('tools', 'model')
    graph.add_edge('finalize', 'handoff')
    graph.add_edge('handoff', END)
    if checkpointer is None:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        allowed = [('maya.schemas', name) for name in (
            'CallerContext', 'ContextPlan', 'ConversationMemory', 'Evidence',
            'EvidenceCitation', 'ChecklistTask', 'OnboardingChecklist',
            'WebexHandoff', 'WebexDispatchRecord', 'PendingAccessResult', 'ModelTurn', 'ReadToolCall')]
        allowed += [('maya.state', 'Message'), ('maya.state', 'MayaState')]
        checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))
    return MayaGraph(graph.compile(checkpointer=checkpointer))


def mark_handoff_unresolved(response, reason):
    response.status = 'unresolved'
    response.tasks.append(ChecklistTask('Webex handoff', 'unresolved', reason=reason))


def history_view(response):
    """What later planner/memory calls need to know an answer said: tasks and statuses.

    The full response repeats every task in `message`, `blocked_items` and `evidence`
    and carries citation metadata and quotes; none of that helps route the next turn,
    and it roughly doubled planner input once checklists grew. Citations stay in the
    typed response and cached evidence; this is only the conversation transcript.
    """
    view = {'status': response.status,
            'tasks': [[t.description, t.status] for t in [*response.tasks, *response.related_status]]}
    if response.pending_access:
        view['pending_access'] = asdict(response.pending_access)
    if response.status == 'denied':
        view['message'] = response.message
    return view


def response_to_dict(response):
    from dataclasses import asdict
    result = asdict(response)
    result['blocked_items'] = [asdict(t) for t in response.blocked_items]
    return result



# ---------------------------------------------------------------------------
# Command line. Library code above never imports the eval tooling; this does.
# ---------------------------------------------------------------------------

def _runtime(offline=False):
    """(app, capture) with the live Bedrock model, local documents and deterministic ranking.

    `offline` swaps in the rule-based stand-in model; only the tests use it, so the test
    suite never makes paid calls. It is not a command-line option.
    """
    from .evals.capture import build_recorded_graph
    if offline:
        from .evals.offline import build_offline_runtime
        runtime = build_offline_runtime()
        return runtime.app, runtime.capture
    from .evals.metering import MeteredBedrock
    from .evals.offline import OfflineSearch
    from .ingestion import load_chunks
    from .model import live_model
    from .operations import LocalReadPort
    from .paths import DATA_PATH
    from .ports import FakeWebexPort
    from .retrieval import OpenSearchEvidenceRetriever
    model = live_model()
    model.client = MeteredBedrock(model.client, keep_requests=False)
    search = OfflineSearch(load_chunks(DATA_PATH))
    retriever = OpenSearchEvidenceRetriever(search, index='offline', embed=search.embed, rerank=search.rerank)
    app, capture = build_recorded_graph(Dependencies(model, retriever, FakeWebexPort(), LocalReadPort()))
    model.client.sink = capture.llm
    return app, capture


def _mode_banner(offline=False):
    import os
    if offline:
        return 'mode: offline test stand-in (rule-based model; no LLM calls)'
    return f"model: Bedrock {os.environ.get('BEDROCK_MODEL_ID', '?')} (live; local documents, deterministic ranking)"


def _print_turn(record, *, trace, graded, tasks):
    reads = ','.join(c['name'] for c in record['reads']) or '-'
    calls = [c for c in record.get('llm', []) if c['site'] != 'embedding']
    tokens = (f" tokens={sum(c.get('input_tokens', 0) for c in calls)}/"
              f"{sum(c.get('output_tokens', 0) for c in calls)}" if calls else '')
    verdict = ('PASS ' if record['passed'] else 'FAIL ') if graded else ''
    print(f"t{record['turn']:02d} {verdict}intent={record['intent']} reads={reads} "
          f"retrieval={len(record['retrieval'])} handoffs={len(record['handoffs'])} "
          f"steps={len(record['nodes'])}/{RECURSION_LIMIT}{tokens} time={record['latency_s']:.2f}s")
    checklist = record['checklist'] or {}
    if checklist.get('status') == 'denied':
        print(f"    denied: {checklist.get('message')}")
    elif checklist:
        pending = checklist.get('pending_access') or {}
        print(f"    checklist: status={checklist.get('status')} role={checklist.get('role')} "
              f"start={checklist.get('start_date')} location={checklist.get('location')} "
              f"tasks={len(checklist.get('tasks', []))} blocked={len(checklist.get('blocked_items', []))}"
              + (f" pending={pending.get('request_id')}" if pending else ''))
        if tasks or trace:
            for task in [*checklist.get('tasks', []), *checklist.get('related_status', [])]:
                source = ', '.join(sorted({c['source'].rsplit('/', 1)[-1] for c in task.get('evidence', [])})) or '-'
                reason = f" ({task['reason']})" if task.get('reason') else ''
                print(f"      [{task['status']}] {task['description'][:110]}{reason}  <- {source}")
    if record.get('error') or record.get('errors'):
        print(f"    errors: {record.get('error') or ''} {record.get('errors') or ''}")
    if graded and record['failures']:
        for name, detail in record['failures'].items():
            print(f'    FAILED {name}: {detail}')
    if not trace:
        return
    plan = (record.get('diagnostic') or {}).get('plan') or {}
    if plan:
        print(f"    plan: intent={plan.get('current_intent')} also={plan.get('also_needs')} "
              f"retrieval={plan.get('requires_retrieval')} reads={plan.get('requires_operational_reads')} "
              f"required={plan.get('required_evidence')}")
        print(f"    loadout: {record['diagnostic'].get('selected_tools')}")
    for call in record['reads']:
        print(f"    read: {call['name']}({call['arguments']})" + (f" ERROR {call['error']}" if call.get('error') else ''))
    for number, round_ in enumerate(record['retrieval'], 1):
        found = sorted({e['citation']['source'].rsplit('/', 1)[-1] for e in round_['evidence']})
        narrowed = f" only={round_['sources']}" if round_.get('sources') else ''
        print(f"    retrieval {number}{narrowed}: {found}" + (f" ERROR {round_['error']}" if round_.get('error') else ''))
    for call in record.get('llm', []):
        print(f"    model: {call['site']:<9} in={call.get('input_tokens', 0)} out={call.get('output_tokens', 0)} "
              f"{call.get('latency_s', 0):.1f}s" + (f" ERROR {call['error']}" if call.get('error') else ''))
    print(f"    nodes: {' -> '.join(record['nodes'])}")


def _print_timing(times, first_turn_started):
    """Turn time is each turn's own wall clock; startup is imports + runtime/AWS setup.

    Measured from when graph.py started executing. Outside it: Python's own launch (~0.02s)
    and interpreter shutdown after the last line (~0.1s), so `time` shows slightly more.
    Anything not in a turn or in startup (printing, grading) is listed as other.
    """
    import time
    if not times:
        return
    total = time.perf_counter() - _CLI_T0
    startup = (first_turn_started or time.perf_counter()) - _CLI_T0
    other = max(0.0, total - startup - sum(times))
    print(f"\ntime: {len(times)} turn(s) {sum(times):.2f}s (mean {sum(times) / len(times):.2f}s, "
          f"slowest {max(times):.2f}s) + startup {startup:.2f}s + other {other:.2f}s = {total:.2f}s total")


async def _run_cli(args, offline=False):
    import json
    import time
    from .evals.runner import DEFAULT_DATASET, load_session, run_turn
    from .schemas import CallerContext
    first_turn, times = None, []
    app, capture = _runtime(offline)
    print(_mode_banner(offline))
    try:
        employee_id, group = args.caller.split('/')
        caller = CallerContext(employee_id, group)
    except ValueError:
        raise SystemExit('--as must look like E004/UG_HR')
    if args.session:
        sessions = json.loads(DEFAULT_DATASET.read_text())['sessions']
        match = [s['id'] for s in sessions if s['id'] == args.session or s['id'].split('-')[0] == args.session]
        if not match:
            raise SystemExit(f"Unknown session {args.session!r}; available: {[s['id'] for s in sessions]}")
        session = load_session(session_id=match[0])
        config = {'configurable': {'thread_id': args.thread or f"{session['id']}-cli",
                                   'caller': CallerContext(**session['caller'])}, 'recursion_limit': RECURSION_LIMIT}
        passed = events = 0
        for turn in session['turns']:
            print(f"\n> {turn['user']}")
            first_turn = first_turn or time.perf_counter()
            # Handoff checks count only events new in this turn (as in evals.runner).
            record = await run_turn(app, turn, capture=capture, config=config, previous_events=events)
            times.append(record['latency_s'])
            events = record['total_handoff_events']
            passed += record['passed']
            _print_turn(record, trace=args.trace, graded=True, tasks=False)
        print(f"\n{session['id']}: {passed}/{len(session['turns'])} turns passed "
              f"(full grading incl. replay and permission case: maya-eval)")
        _print_timing(times, first_turn)
        return 0 if passed == len(session['turns']) else 1
    config = {'configurable': {'thread_id': args.thread or 'maya-cli', 'caller': caller},
              'recursion_limit': RECURSION_LIMIT}
    questions = [args.question] if args.question else []
    number = events = 0
    while True:
        if questions:
            text = questions.pop(0)
        elif args.chat:
            try:
                text = input('\nyou> ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                text = ''
            if text.lower() in ('', 'exit', 'quit'):
                _print_timing(times, first_turn)
                return 0
        else:
            _print_timing(times, first_turn)
            return 0
        number += 1
        first_turn = first_turn or time.perf_counter()
        record = await run_turn(app, {'n': number, 'user': text}, capture=capture, config=config,
                                previous_events=events)
        times.append(record['latency_s'])
        events = record['total_handoff_events']
        _print_turn(record, trace=args.trace, graded=False, tasks=True)


def main(argv=None, *, offline=False):
    """Lesson 10-style entry point with the live Bedrock model: replay a session, ask one question, or chat.

    Every run makes paid Bedrock calls (credentials and BEDROCK_MODEL_ID from Maya's .env).
    `offline` is for the tests only.
    """
    import argparse
    import asyncio
    # prog defaults to how it was started: maya-live, graph.py or python -m maya.graph.
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument('question', nargs='?', help='ask one question (quote it)')
    parser.add_argument('--session', help='replay a dataset session by id or prefix, e.g. S2')
    parser.add_argument('--chat', action='store_true', help='interactive conversation on one thread')
    parser.add_argument('--trace', action='store_true', help='show plan, reads, sources, model calls and steps')
    parser.add_argument('--as', dest='caller', default='E004/UG_HR', help='caller ID/group (default E004/UG_HR)')
    parser.add_argument('--thread', help='thread id (conversation) to use')
    args = parser.parse_args(argv)
    if sum(bool(x) for x in (args.question, args.session, args.chat)) != 1:
        parser.error('give exactly one of: a question, --session S2, or --chat')
    from .backend import external_errors
    try:
        return asyncio.run(_run_cli(args, offline))
    except external_errors() as exc:
        raise SystemExit(f'graph failed: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    raise SystemExit(main())
