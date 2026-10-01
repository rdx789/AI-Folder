"""Static reachability proof for every adapted S2 intent alternative."""
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/matrix.py): relative imports need the package, so
    # re-run this file as the installed module `maya.matrix`.
    import runpy
    try:
        runpy.run_module('maya.matrix', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit
import json
from pathlib import Path
from .policy import READ_TOOLS, select_tools
from .schemas import ContextPlan


def check_s2_matrix(path=None):
    data = json.loads(Path(path or Path(__file__).with_name('evals') / 'data' / 's2_maya.json').read_text())
    rows = []
    for turn in data['turns']:
        intents = (turn['expected_intent'] or 'recall').split('|')
        for intent in intents:
            plan = ContextPlan(intent, requires_operational_reads=bool(turn['expected_read_tools']))
            selected = select_tools(plan)
            missing = set(turn['expected_read_tools']) - set(selected)
            if missing:
                raise AssertionError(f"S2 turn {turn['n']} {intent}: unreachable {sorted(missing)}")
            for fallback in (False, True):
                names = select_tools(plan, fallback=fallback)
                if not set(names) <= READ_TOOLS or 'create_access_request' in names:
                    raise AssertionError('Write tool exposed')
                if not turn['should_use_tools'] and names:
                    raise AssertionError('Recall/action-only turn exposes read tools')
            rows.append({'turn': turn['n'], 'intent': intent, 'required': turn['expected_read_tools'],
                         'selected': list(selected), 'retrieval': turn['requires_retrieval'],
                         'handoff': turn['expected_handoff']})
    return rows


if __name__ == '__main__':
    print(json.dumps(check_s2_matrix(), indent=2))
