"""Distillation policy; acceptance checks preserve values, not just counts."""

from .schemas import ConversationMemory
from .state import MayaState
import re

FRESH_MESSAGE_LIMIT = 12
APPROX_TOKEN_LIMIT = 2500


def needs_distillation(state: MayaState) -> bool:
    fresh = state.messages[state.distilled_upto:]
    return len(fresh) >= FRESH_MESSAGE_LIMIT or sum(len(m.content) for m in fresh) / 4 >= APPROX_TOKEN_LIMIT


def memory_is_safe(previous: ConversationMemory, candidate: ConversationMemory, chunk=()) -> bool:
    """Compression cannot silently revise facts or remove constraints/closed topics.

    Explicit corrections/reopened topics must be applied by a future validated
    turn-update step before compression. This check is not semantic verification.
    """
    # SDD checks directives in the block being folded as well as rules already
    # in memory. Otherwise the first fold can silently lose a new user rule.
    constraints = ' '.join(candidate.active_constraints.values()).casefold()
    stems = lambda text: {w[:4] for w in re.findall(r'[a-z]{4,}', text.casefold())
                           if w not in {'that', 'this', 'with', 'until', 'unless', 'must', 'please'}}
    for message in chunk:
        if message.role != 'user':
            continue
        for clause in re.split(r'(?<=[.!?;:])\s+|\n+|,\s*(?:and|but|also)\s+', message.content):
            if re.search(r"\b(do not|don't|until|unless|without|approval|must|not allowed)\b", clause, re.I):
                if len(stems(clause) & stems(constraints)) < 2:
                    return False
    return (
        all(candidate.important_facts.get(k) == v for k, v in previous.important_facts.items())
        and all(candidate.active_constraints.get(k) == v for k, v in previous.active_constraints.items())
        and set(previous.decisions) <= set(candidate.decisions)
        and set(previous.unresolved_items) <= set(candidate.unresolved_items)
        and set(previous.closed_topics) <= set(candidate.closed_topics)
    )


def distillation_boundary(state: MayaState) -> int:
    """Fold only complete older turns; keep the newest two user turns intact."""
    starts = [i for i, m in enumerate(state.messages) if m.role == 'user']
    return starts[-2] if len(starts) > 2 else state.distilled_upto


def visible_history(state: MayaState, messages):
    """Remove a closed tangent's entire exchange from model context."""
    output, suppress = [], False
    for message in messages:
        if message.role == 'user':
            suppress = any(topic.lower() in message.content.lower()
                           for topic in state.memory.closed_topics)
        if not suppress:
            output.append(message)
    return output
