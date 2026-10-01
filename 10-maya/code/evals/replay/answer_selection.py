"""Replay: Maya's answer-selection call (turn 8 restated the Webex blocker as unresolved).

Target: run1#8 (failed live) and run2#8+prose (a passing input plus the failing prose
fact). Guards: every other selection turn of two live runs, scored by S2 labels.

    .venv/bin/python ~/.claude/skills/prompt-replay-test/scripts/replay.py \
        code/evals/replay/answer_selection.py --repeats 3
"""
import copy

from maya.answer_context import (AnswerSelection, candidate_catalogue, focused_evidence,
                                 required_read_handles, resolve_selection)
from maya.model import BedrockMayaModel, live_model
from maya.operations import tool_schemas
from maya.policy import grounded_facts, necessary_reads
from maya.schemas import ContextPlan, ConversationMemory, Evidence, EvidenceCitation
from maya.state import Message

from maya.paths import RESULTS_DIR
INPUTS = {'s2': str(RESULTS_DIR / 'replay' / 'cases.pkl')}

# (label, index in cases.pkl, expected, kind); expected = S2 required_facts.
LABELS = ['run1#2', 'run1#3', 'run1#4', 'run1#5', 'run1#6', 'run1#7', 'run1#8', 'run1#9', 'run1#11',
          'run1#12', 'run2#2', 'run2#3', 'run2#4', 'run2#5', 'run2#6', 'run2#7', 'run2#8', 'run2#9',
          'run2#11', 'run2#12', 'run2#8+prose']
TARGETS = {'run1#8', 'run2#8+prose'}
CASES = [('s2', i, {'label': label}, 'target' if label in TARGETS else 'guard')
         for i, label in enumerate(LABELS)]

OLD = 'If evidence is genuinely missing, unresolved_items describes what could not be verified.'
NEW = ('unresolved_items is ONLY for a needed fact that NO catalogue statement supports. If a\n'
       'catalogue statement states a blocker, seat count, capacity limit, threshold or\n'
       'requirement, select its id; never restate, paraphrase or summarise catalogue content,\n'
       'facts or prior answers in unresolved_items. facts are carried context, not evidence.')


def variant(*, prompt=False, grounded=False, backstop=False):
    def apply(ctx):
        ctx = copy.deepcopy(ctx)
        ctx['edits'] = [(OLD, NEW)] if prompt else []
        ctx['grounded'], ctx['backstop'] = grounded, backstop
        return ctx
    return apply


VARIANTS = {
    'current': variant(),
    'prompt': variant(prompt=True),
    'facts': variant(grounded=True),
    'backstop': variant(backstop=True),
    'prompt+facts': variant(prompt=True, grounded=True),
    'all': variant(prompt=True, grounded=True, backstop=True),
}

_base = live_model()


class EditedClient:
    """Applies the variant's system-prompt edits; fails loudly if the text drifted."""
    def __init__(self, client, edits):
        self.client, self.edits = client, edits

    def converse(self, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        text = kwargs['system'][0]['text']
        for old, new in self.edits:
            if old not in text:
                raise ValueError(f'prompt text not found: {old[:50]!r}')
            text = text.replace(old, new)
        kwargs['system'][0]['text'] = text
        return self.client.converse(**kwargs)


class Recording(BedrockMayaModel):
    """Keeps the raw AnswerSelection so the code backstop can be scored on it."""
    async def _structured(self, output_type, instruction, payload, *, tools=None):
        result = await super()._structured(output_type, instruction, payload, tools=tools)
        if output_type is AnswerSelection:
            self.selection = result
        return result


async def call(ctx):
    plan = ContextPlan(**copy.deepcopy(ctx['plan']))
    if ctx['grounded']:
        plan.relevant_facts = grounded_facts(plan.relevant_facts, user_texts=ctx['user_texts'],
                                             memory_values=ctx['memory_values'])
    evidence = tuple(Evidence(e['text'], EvidenceCitation(**{**e['citation'],
                     'audience': tuple(e['citation']['audience'])})) for e in ctx['evidence'])
    history = [Message('user', ctx['newest'])]
    if ctx['completed']:
        history.append(Message('assistant', '', tuple(ctx['completed'])))
    model = Recording(EditedClient(_base.client, ctx['edits']), model_id=_base.model_id)
    turn = await model.respond(newest_message=ctx['newest'], plan=plan, memory=ConversationMemory(),
                               history=history, evidence=evidence,
                               tools=tool_schemas(ctx['selected_tools'] if ctx.get('tools_available', True) else ()),
                               previous_checklist=None)
    tasks = turn.tasks
    selection = getattr(model, 'selection', None)
    if ctx['backstop'] and selection is not None:
        focused = focused_evidence(evidence, plan, ctx['newest'])
        candidates = candidate_catalogue(focused)
        must = required_read_handles(candidates, required_reads=necessary_reads(plan, ctx['newest']),
                                     completed=set(ctx['completed']))
        selection.candidate_ids = list(dict.fromkeys([*selection.candidate_ids, *must]))
        tasks = resolve_selection(selection, candidates, focused)
    visible = [str(v) for v in ctx['state_fields'].values() if v] + list(ctx['active_constraints'])
    visible += [f"{t['description']} {t['status']}" for t in ctx['known_tasks']]
    visible += [f'{t.description} {t.status} {t.reason or ""}' for t in tasks]
    return {'visible': '\n'.join(visible).casefold(), 'required': ctx['required_facts'],
            'model_unresolved': sum(t.reason == 'Model identified missing evidence' for t in tasks),
            'selected': len(selection.candidate_ids) if selection else None, 'tool_calls': len(turn.tool_calls)}


def score(case, output):
    """Same rule as evals/checks.py: every required fact (any '|' alternative) is displayed."""
    return all(any(alt.casefold() in output['visible'] for alt in fact.split('|'))
               for fact in output['required'])
