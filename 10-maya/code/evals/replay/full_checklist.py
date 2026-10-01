"""Replay: does the answer model cover every section of a full onboarding checklist?

Target: Sara's first request (live model selected only database rows). Guards: the ten
S2 selection turns of the same run, scored by their S2 labels.

    .venv/bin/python ~/.claude/skills/prompt-replay-test/scripts/replay.py \
        code/evals/replay/full_checklist.py --repeats 5
"""
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import answer_selection as base  # noqa: E402  (shared call/score machinery)

from maya.paths import RESULTS_DIR
INPUTS = {'s2': str(RESULTS_DIR / 'replay' / 'full_checklist_cases.pkl')}
LABELS = [c['label'] for c in pickle.load(open(INPUTS['s2'], 'rb'))]
CASES = [('s2', i, {'label': label}, 'target' if label.endswith('#first') else 'guard')
         for i, label in enumerate(LABELS)]

ANCHOR = 'For a summary include cited requirements/thresholds plus open tasks and blockers.\n'
DETAILED = ANCHOR + (
    'For a full onboarding checklist (the newest message asks for systems, equipment, policy\n'
    'acknowledgements and/or blockers), cover EVERY requested section from the catalogue:\n'
    "required systems and the role's baseline bundle; equipment entitlements; each policy rule or\n"
    'acknowledgement the employee, HR or IT must satisfy (signed-offer rule, equipment\n'
    'responsibilities, acceptable use, access approvals); every blocker with its capacity\n'
    "evidence. Database rows alone are not a complete checklist. Skip rules that do not apply\n"
    "to this employee's role (e.g. contractor-only or admin-only access).\n")
SHORT = ANCHOR + (
    'A full onboarding checklist selects statements for every requested section (systems,\n'
    'equipment, policy rules/acknowledgements, approvals, blockers), not only database rows.\n')


def with_edit(new):
    def apply(ctx):
        ctx = base.variant()(ctx)
        ctx['edits'] = [(ANCHOR, new)]
        return ctx
    return apply


SCOPED = ANCHOR + (
    'ONLY when the newest message asks for a full onboarding checklist (systems, equipment,\n'
    'policy acknowledgements, blockers): cover each requested section with the statements that\n'
    "APPLY TO THIS EMPLOYEE: required systems and the role's baseline bundle; equipment\n"
    'entitlements; the policy rules and acknowledgements she, HR or IT must satisfy (signed-offer\n'
    'rule, equipment responsibilities, acceptable use, standard access approval); each blocker\n'
    'with its capacity evidence. Database rows alone are not a complete checklist. Do NOT select\n'
    'rules for other audiences or cases: contractors, privileged/admin or AWS access, personal-\n'
    'device exceptions, offboarding, or generic process notes.\n')
SCOPED_SHORT = ANCHOR + (
    'ONLY for a full onboarding checklist request: select, per requested section, the statements\n'
    'that apply to this employee (systems, equipment, her policy acknowledgements, standard access\n'
    'approval, blockers with capacity evidence); never rules for contractors, admin/AWS access,\n'
    'exceptions or other audiences.\n')

# Round 1 (before the code scope filter): detailed 5/5 but 3-9 irrelevant picks and guard
# selection 6.9 -> 10.9; short 2/5; scoped_short 0/5; scoped 5/5 with 1-3 irrelevant.
# Round 2 (with the code scope filter): scoped 5/5, 0 irrelevant, guard selection 6.7 vs 7.9.
# The scoped rule now ships in model.ANSWER_INSTRUCTIONS, so 'current' is the shipped prompt.
VARIANTS = {'current': base.variant()}

# Only statements whose SUBJECT is another audience; "employees and contractors must not..."
# applies to employees too.
IRRELEVANT = re.compile(r'^(github|aws admin|contractors?|privileged or admin)\b', re.I | re.M)


async def call(ctx):
    output = await base.call(ctx)
    output['irrelevant'] = len(IRRELEVANT.findall(output['visible']))
    return output


score = base.score
