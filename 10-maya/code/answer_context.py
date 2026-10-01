"""Focused source context and extractive choices; no model-authored citations."""
from dataclasses import dataclass, field
import hashlib
import json
import re

from .grounding import normalize, rank, status_is_supported, validate_tasks
from .policy import full_checklist_request
from .schemas import ChecklistTask, ReadToolCall


@dataclass
class AnswerSelection:
    tool_calls: list[ReadToolCall] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)


def focused_evidence(evidence, plan, newest):
    """Keep the complete current source passages, instead of all past topics."""
    # A full checklist spans every section, so it keeps every retrieved source.
    if plan.current_intent in ('recall', 'access_request') or full_checklist_request(newest):
        return list(evidence)
    sources = set(plan.required_evidence)
    operational_groups = {
        'employee_lookup': {'get_employee'},
        'onboarding_status': {'list_onboarding_tasks'},
        'ticket_status': {'list_employee_tickets'},
        'equipment_request': {'check_asset_inventory'},
        'subscription_review': {'check_software_subscription'},
    }
    operational = set().union(*(operational_groups.get(intent,set())
                                for intent in (plan.current_intent,*plan.also_needs)))
    documents = {
        'equipment_request': ('equipment',),
        'subscription_review': ('webex',),
        'policy_question': ('policy', 'memo'),
    }.get(plan.current_intent, ())
    result = [e for e in evidence if (
        e.citation.source.rsplit('/', 1)[-1] in sources
        or e.citation.source.removeprefix('novaops/') in operational
        or (not sources and e.citation.collection != 'operational'
            and (plan.current_intent == 'onboarding_status' or any(
                word in e.citation.source for word in documents))))]
    return sorted(result or list(evidence), key=rank, reverse=True)


def _document_parts(text):
    # Ingested Markdown can be flattened. Split only at source boundaries; all
    # returned strings remain exact substrings, including monetary thresholds.
    for part in re.split(r'(?<=[.!?])\s+|\n+|\s+(?=- |## )', text):
        part = part.strip().removeprefix('- ').strip()
        if part and not part.startswith('#') and not re.match(
                r'^(Owner|Effective date|Last updated|Date):', part):
            yield part
        elif part.startswith('#'):
            # Separate flattened title/metadata from the first body sentence.
            match = re.search(r'\b\d{4}-\d{2}-\d{2}\s+(.*)', part)
            body = match[1].strip() if match else heading_body(part)
            if body and not body.startswith('#'):
                yield body


SENTENCE_STARTERS = frozenset({'If', 'When', 'For', 'Do', 'All', 'Each', 'Any', 'No', 'The',
                               'A', 'An', 'This', 'These', 'Being', 'Only'})


def heading_body(part):
    """'## External sharing External sharing of data must...' -> the sentence after the heading.

    Chunks are flattened, so a section heading and its first sentence share a part.
    The body starts at the first word that opens a sentence: capitalised and followed
    by a lowercase word, or a sentence starter ("If Webex access..." keeps its "If").
    Returns None when no clear boundary exists within a short heading.
    """
    words = part.lstrip('#').split()
    for k in range(1, min(8, len(words) - 1)):
        first, second = words[k], words[k + 1]
        if first[:1].isupper() and (first in SENTENCE_STARTERS or second[:1].islower()):
            return ' '.join(words[k:])
    return None


def candidate_catalogue(evidence):
    """Build statements without deciding what answer the user should receive."""
    candidates = {}
    for passage in evidence:
        citation = passage.citation
        tasks = []
        if citation.collection == 'operational':
            try:
                rows = json.loads(passage.text)
            except (ValueError, TypeError):
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                quote = json.dumps(row, sort_keys=True)
                status = {'completed': 'complete', 'pending': 'pending', 'open': 'pending',
                          'planned': 'proposed', 'available': 'proposed', 'blocked': 'blocked'}.get(row.get('status'))
                if citation.source == 'novaops/check_software_subscription':
                    used, limit = row.get('active_seats'), row.get('seat_limit')
                    status = 'blocked' if type(used) is int and type(limit) is int and used >= limit else None
                if status is None:
                    continue
                # Inventory/subscription rows retain asset IDs, model and counts.
                description = row.get('description') or quote
                key = str(row.get('task_id') or row.get('asset_id') or row.get('ticket_id')
                          or row.get('system_name') or citation.source)
                tasks.append(ChecklistTask(description, status, (citation,), key, (quote,)))
        else:
            for part in _document_parts(passage.text):
                key = citation.source + ':' + hashlib.sha256(normalize(part).encode()).hexdigest()[:16]
                # Comparable monetary approval requirements share a key: newer
                # explicit dates win; equal-ranked disagreement stays unresolved.
                if re.search(r'above.*(?:USD|\$).*Finance approval', part, re.I):
                    key = 'guidance:finance_threshold'
                for status in ('blocked', 'pending', 'proposed', 'complete'):
                    task = ChecklistTask(part, status, (citation,), key, (part,))
                    if status_is_supported(task, (passage,)):
                        tasks.append(task)
                        break
        for task in validate_tasks(tasks, (passage,)):
            if task.status == 'unresolved':
                continue
            handle = hashlib.sha256((citation.chunk_id + task.task_id + task.description).encode()).hexdigest()[:20]
            candidates[handle] = task
    return candidates


def catalogue_payload(candidates):
    return [{'id': handle, 'statement': task.description, 'status': task.status,
             'source': task.evidence[0].source, 'updated_at': task.evidence[0].updated_at,
             'subject_employee_id': task.evidence[0].subject_employee_id,
             'support_quote': task.support_quotes[0]} for handle, task in candidates.items()]


def resolve_selection(selection, candidates, evidence):
    if any(handle not in candidates for handle in selection.candidate_ids):
        raise ValueError('Model selected evidence outside the supplied catalogue')
    keys = {candidates[handle].task_id for handle in selection.candidate_ids}
    # Include competitors automatically: the model cannot hide an older/newer
    # version sharing a requirement key by choosing just the favourable citation.
    tasks = validate_tasks([task for task in candidates.values() if task.task_id in keys], evidence)
    tasks += [ChecklistTask(item, 'unresolved', reason='Model identified missing evidence')
              for item in selection.unresolved_items]
    return tasks


def planner_transcript(history):
    """Keep current roles clear and clip raw results like the SDD projection."""
    return [{'role': m.role, 'content': m.content[:400] if m.role == 'tool' else m.content,
             'tool_calls': m.tool_calls} for m in history]


def required_read_handles(candidates, *, required_reads, completed):
    """Rows of this turn's application-required reads always reach the checklist.

    The planner chose the source and the graph executed the read; which of its
    status rows answer the question is not a model judgement to skip.
    """
    sources = {'novaops/' + name for name in required_reads if name in completed}
    return [handle for handle, task in candidates.items()
            if task.evidence and task.evidence[0].source in sources]


def scoped_catalogue(candidates, newest, evidence):
    """A full checklist is for this employee's baseline onboarding.

    Drops rules whose subject is another audience: contractor rules unless the
    authorized record says contractor, privileged/admin/AWS-admin rules unless the
    request asks for them. Only the statement's subject counts, so "what employees
    and contractors must not do" still applies.
    """
    employment = {row.get('employment_type') for item in evidence
                  if item.citation.source == 'novaops/get_employee'
                  for row in _rows(item.text)}
    wants_admin = re.search(r'\b(admin|privileged|AWS)\b', newest, re.I)

    def in_scope(task):
        if re.match(r'contractors?\b', task.description, re.I):
            return 'Contractor' in employment
        if re.match(r'(privileged|admin|aws admin)\b', task.description, re.I):
            return bool(wants_admin)
        return True
    return {handle: task for handle, task in candidates.items() if in_scope(task)}


def _rows(text):
    try:
        rows = json.loads(text)
    except (ValueError, TypeError):
        return []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


# Documents whose requirements a full checklist lists completely, not as a sample.
ACKNOWLEDGEMENT_SOURCES = ('onboarding_policy.md', 'equipment_responsibility_form.md',
                           'acceptable_use_policy.md')


def acknowledgement_handles(candidates):
    """Every in-scope requirement of the acknowledgement documents (run after scoping)."""
    return [handle for handle, task in candidates.items()
            if task.status in ('proposed', 'pending') and task.evidence
            and task.evidence[0].source.rsplit('/', 1)[-1] in ACKNOWLEDGEMENT_SOURCES]
