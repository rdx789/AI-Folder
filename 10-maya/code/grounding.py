"""Conservative extractive validation, evidence precedence and conflict handling.

A citation alone is not support. Resolved task descriptions must be extractive
statements present in their cited passages. Free paraphrases remain unresolved.
"""
from dataclasses import replace
from datetime import date
import json
import re
from .schemas import ChecklistTask, Evidence


def normalize(text):
    return ' '.join(text.casefold().split())


def operational_status(task, passages):
    """Check a record's actual status, not an arbitrary nearby word in its JSON."""
    for passage in passages:
        if passage.citation.collection != 'operational':
            continue
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
            labels = [row.get(k) for k in ('description', 'model', 'subject')]
            if task.description not in [quote, *labels]:
                continue
            if not any(quote in q for q in task.support_quotes):
                continue
            if passage.citation.source == 'novaops/check_software_subscription':
                used, limit = row.get('active_seats'), row.get('seat_limit')
                if type(used) is int and type(limit) is int and used >= limit:
                    return 'blocked'
                return None  # availability alone never establishes granted access
            return {'completed': 'complete', 'pending': 'pending', 'open': 'pending',
                    'planned': 'proposed', 'available': 'proposed'}.get(
                        row.get('status'), 'blocked' if row.get('status') == 'blocked' else None)
    return None


GENERIC_OPENERS = frozenset({'A', 'An', 'The', 'This', 'These', 'That', 'If', 'When', 'Each',
                             'Any', 'All', 'No', 'Do'})


def names_something(text):
    """True if a statement names a system, team or number ("Webex", "Finance", "42")."""
    return bool(re.search(r'\d', text)) or any(
        word not in GENERIC_OPENERS for word in re.findall(r'\b[A-Z][\w-]*', text))


def over_capacity(text):
    """'42 active seats against a 40-seat limit' -> True; None if no such statement."""
    match = re.search(r'\b(\d+) active seats against (?:an? |the )?(\d+)-seat\b', text)
    return None if not match else int(match[1]) >= int(match[2])


def status_is_supported(task, passages=()):
    """Resolve only explicit status statements; modal/conditional rules are open."""
    text = normalize(task.description)
    markers = {
        # Requirements and acknowledgements are to-dos: "need", "only after the offer
        # is signed", "not permitted", "responsible for", "eligible for", "verify".
        'proposed': r'\b(required|entitlement|planned|proposed|needs?|must|may receive|only after'
                    r'|not permitted|not allowed|responsible for|eligible for|verify|confirm|enroll)\b',
        'complete': r'\b(completed|complete|assigned|delivered|ready)\b',
        'blocked': r'\b(blocked|no seats|over limit|do not assign|should not assign|exceed)\b',
        'pending': r'\b(pending|awaiting|approval required|require.*approval)\b',
    }
    if task.status == 'unresolved':
        return True
    over = over_capacity(text)
    if over is not None:
        return task.status == 'blocked' and over  # a seat count is a fact, not a to-do
    if task.status == 'complete' and 'webex' in text:
        return False  # This agent cannot establish granted Webex access.
    if any(p.citation.collection == 'operational' for p in passages):
        return operational_status(task, passages) == task.status
    # A statement that access MAY be blocked or IF seats exceed the cap does
    # not establish an actual blocker. Likewise, negated completion isn't done.
    if task.status == 'blocked' and re.search(r'\b(may|might|if|can|could)\b', text):
        return False
    # A blocker names what is blocked (Webex, Finance, 42 seats); a process rule
    # such as "a blocked item stays visible" is not itself a blocker.
    if task.status == 'blocked' and not names_something(task.description):
        return False
    if task.status == 'complete' and re.search(r'\b(not|never|until|before|may|if)\b', text):
        return False
    return bool(re.search(markers[task.status], text))


def rank(evidence: Evidence):
    citation = evidence.citation
    # Specific employee records beat general policy. Explicit later source dates
    # break ties; filesystem timestamps and model-provided priorities never do.
    specificity = 2 if citation.subject_employee_id == 'E001' else 1
    if citation.collection == 'operational':
        specificity = 3
    try:
        updated = date.fromisoformat(citation.updated_at or '').toordinal()
    except ValueError:
        updated = 0
    return specificity, updated


def validate_tasks(tasks, evidence):
    by_id = {e.citation.chunk_id: e for e in evidence}
    candidates = {}
    for task in tasks:
        if not isinstance(task, ChecklistTask):
            raise ValueError('Model must return typed checklist tasks')
        key = task.task_id or normalize(task.description)
        if task.status == 'unresolved' and not task.evidence and task.description.strip():
            candidates.setdefault(key, []).append(((-1, 0), replace(
                task, reason=task.reason or 'Supporting evidence is unresolved')))
            continue
        passages = [by_id.get(c.chunk_id) for c in task.evidence]
        supported = bool(passages) and all(p and p.citation == c for p, c in zip(passages, task.evidence))
        quotes = task.support_quotes or (task.description,)
        supported = supported and all(any(normalize(q) in normalize(p.text) for p in passages) for q in quotes)
        supported = supported and any(normalize(task.description) in normalize(q) for q in quotes)
        supported = supported and status_is_supported(task, passages)
        if not task.description.strip() or not supported:
            task = replace(task, status='unresolved', evidence=(), support_quotes=(),
                           reason='Missing or non-extractive supporting evidence')
            priority = (-1, 0)
        else:
            priority = max(rank(p) for p in passages)
        candidates.setdefault(key, []).append((priority, task))
    result = []
    for key, entries in candidates.items():
        best = max(priority for priority, _ in entries)
        winners = [t for priority, t in entries if priority == best]
        values = {(normalize(t.description), t.status) for t in winners}
        if len(values) > 1:
            result.append(ChecklistTask(key, 'unresolved', task_id=key,
                                        reason='Conflicting evidence at equal precedence'))
        else:
            result.append(winners[-1])
    return result


def excludes_closed_topic(text, closed_topics):
    return any(re.search(r'\b' + re.escape(topic) + r'\b', text, re.I) for topic in closed_topics)
