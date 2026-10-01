"""Rebuild the answer-selection inputs of saved live S2 runs for prompt replay.

Each record holds the turn's plan, selected tools, executed reads and cached evidence,
which are exactly the inputs of that turn's final `respond` selection call.

    .venv/bin/python code/evals/replay/build_cases.py OUT.pkl LABEL=report.json [LABEL=report.json ...]
"""
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/evals/replay/build_cases.py): relative imports need the package, so
    # re-run this file as the installed module `maya.evals.replay.build_cases`.
    import runpy
    try:
        runpy.run_module('maya.evals.replay.build_cases', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit
import json
import pickle

SELECTION_TURNS = {2, 3, 4, 5, 6, 7, 8, 9, 11, 12}  # 1 and 10 bypass the model
PROSE = 'The Webex license request is blocked due to a seat limit.'


def cases_from(label, report, dataset):
    users = {t['n']: t['user'] for t in dataset['turns']}
    facts = {t['n']: t.get('required_facts', []) for t in dataset['turns']}
    out = []
    for record in report['records']:
        n = record['turn']
        if n not in SELECTION_TURNS or not record['diagnostic']:
            continue
        out.append(case(f'{label}#{n}', record, [users[i] for i in range(1, n + 1)],
                        facts[n], known=n == 12))
    first = report.get('first_request')
    if first and first.get('diagnostic'):
        from ..runner import FIRST_REQUEST
        out.append(case(f'{label}#first', first, [first['user']], FIRST_REQUEST['required_facts']))
    return out


def case(name, record, user_texts, required_facts, known=False):
    diagnostic = record['diagnostic']
    memory = diagnostic['memory']
    return {
        'label': name, 'turn': record['turn'], 'newest': record['user'],
        'plan': diagnostic['plan'], 'selected_tools': diagnostic['selected_tools'],
        'completed': [c['name'] for c in record['reads']],
        # The final answer call gets no read tools once both read rounds are used.
        'tools_available': diagnostic.get('tool_rounds', 0) < 2,
        'evidence': record['cached_evidence'],
        'user_texts': user_texts,
        'memory_values': [*memory['important_facts'].values(), *memory['active_constraints'].values()],
        'state_fields': {k: (record['checklist'] or {}).get(k) for k in
                         ('full_name', 'employee_id', 'role', 'start_date', 'location', 'work_mode')},
        'active_constraints': (record['checklist'] or {}).get('active_constraints', []),
        # Recall summaries merge previously known tasks with this selection.
        'known_tasks': [t for t in (record['checklist'] or {}).get('tasks', [])
                        if t.get('reason') != 'Model identified missing evidence'] if known else [],
        'required_facts': required_facts,
    }


def main(argv=None):
    import argparse
    from .. import runner
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', help='where to write the case pickle, e.g. results/replay/cases.pkl')
    parser.add_argument('reports', nargs='+', metavar='LABEL=report.json', help='saved S2 report(s) to rebuild from')
    args = parser.parse_args(argv)
    dataset = runner.load_session()
    cases = []
    for arg in args.reports:
        label, sep, path = arg.partition('=')
        if not sep or not label or not path:
            parser.error(f'expected LABEL=report.json, got {arg!r}')
        try:
            with open(path) as f:
                cases += cases_from(label, json.load(f), dataset)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f'cannot read report {path}: {type(exc).__name__}: {exc}')
    # Hypothesis probe: a passing turn-8 input plus the prose fact seen in the failure.
    for case in list(cases):
        if case['label'].endswith('#8') and PROSE not in case['plan']['relevant_facts']:
            probe = json.loads(json.dumps(case))
            probe['label'] += '+prose'
            probe['plan']['relevant_facts'].append(PROSE)
            cases.append(probe)
    with open(args.output, 'wb') as f:
        pickle.dump(cases, f)
    for i, case in enumerate(cases):
        print(i, case['label'], case['required_facts'])


if __name__ == '__main__':
    main()
