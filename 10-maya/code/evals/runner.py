"""Ordered session replay adapted from Lesson 10 code/evals/runner.py.

Retains its stable-thread turn loop, per-turn exception capture and deterministic
records; replaces lesson message/meter/tool APIs with typed Maya boundaries.
No Langfuse, model judge, paid API or lesson-package import is needed offline.
"""
if __name__ == '__main__' and not __package__:
    # Started by file path (python code/evals/runner.py): relative imports need the package, so
    # re-run this file as the installed module `maya.evals.runner`.
    import runpy
    try:
        runpy.run_module('maya.evals.runner', run_name='__main__', alter_sys=True)
    except ImportError as exc:  # runpy wraps the ModuleNotFoundError for the package
        if 'maya' not in (getattr(exc, 'name', None), getattr(exc.__cause__, 'name', None)):
            raise
        raise SystemExit("maya is not installed for this Python. Use the project's interpreter: "
                         ".venv/bin/python (or `source .venv/bin/activate`), or run ./setup.sh")
    raise SystemExit
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

from ..graph import RECURSION_LIMIT, response_to_dict
from ..matrix import check_s2_matrix
from ..paths import RESULTS_DIR
from ..policy import READ_TOOLS, LOADOUTS, select_tools
from ..schemas import CallerContext, ContextPlan
from .checks import check_turn, failures
from .offline import build_offline_runtime

DEFAULT_DATASET = Path(__file__).with_name('data') / 'sessions.json'
# The homework's headline request, run alone on its own thread (not part of the S2 file).
FIRST_REQUEST = {
    'n': 1, 'kind': 'first_request', 'should_use_tools': True,
    'user': ('Prepare an evidence-backed onboarding checklist for Maya Cohen, a Customer Success '
             'Manager starting 2026-08-01. Include required systems, equipment, policy '
             'acknowledgements, and anything currently blocked.'),
    'expected_tools': [], 'forbidden_tools': ['create_access_request'],
    'required_facts': ['Customer Success Manager', '2026-08-01', 'Israel', 'Okta', 'Salesforce',
                       'Webex', 'SupportDesk', 'laptop', 'monitor', 'dock', 'headset',
                       'offer is signed', 'responsible for', 'least privilege|personal account',
                       'manager approval', '42', 'blocked'],
    # Homework Milestone 2: the successful Maya path must be able to cite these.
    'required_sources': ['maya_cohen_offer_letter.md', 'maya_cohen_onboarding_summary.md',
                         'onboarding_policy.md', 'equipment_policy.md', 'access_management_policy.md',
                         'salesforce_access_request.md', 'webex_license_assignment.md',
                         'laptop_provisioning.md'],
    'cited_sources': ['maya_cohen_offer_letter.md', 'onboarding_policy.md',
                      'equipment_policy.md|laptop_provisioning.md', 'equipment_responsibility_form.md',
                      'webex_license_cleanup_memo.md|check_software_subscription'],
}


def load_session(path=None, session_id='S2-onboarding-maya'):
    data = json.loads(Path(path or DEFAULT_DATASET).read_text())
    session = next((s for s in data['sessions'] if s['id'] == session_id), None)
    if session is None:
        raise ValueError(f'Session not found: {session_id}')
    if [t['n'] for t in session['turns']] != list(range(1, 13)):
        raise ValueError('S2 must contain all twelve ordered turns')
    CallerContext(**session['caller'])
    return session


async def run_turn(app, turn, *, capture, config, previous_events=0, replay=False):
    position = capture.position()
    started = time.perf_counter()
    error, result = None, None
    try:
        result = await app.ainvoke({'newest_message': turn['user']}, config=config)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        # A failed node still leaves a useful checkpoint. Preserve it for
        # diagnosis rather than erasing the plan/loadout on recursion failure.
        try:
            snapshot = await app.compiled.aget_state(config)
            result = {'session': snapshot.values.get('session')}
        except Exception:
            pass
    state = result.get('session') if result else None
    response = result.get('response') if result else None
    record = {
        'turn': turn['n'], 'kind': turn.get('kind','lookup'), 'replay': replay,
        'thread_id': config['configurable']['thread_id'], 'caller': asdict(config['configurable']['caller']),
        'user': turn['user'], 'expected': deepcopy(turn),
        'intent': state.plan.current_intent if state and state.plan else None,
        'answer': response.message if response else '',
        'checklist': response_to_dict(response) if response else None,
        'error': error, 'errors': list(state.errors) if state else [],
        'nodes': list(state.node_visits) if state else [],
        'cached_evidence': [asdict(e) for e in state.evidence] if state else [],
        'handoff_events': [asdict(h) for h in state.handoff_events[previous_events:]] if state else [],
        'total_handoff_events': len(state.handoff_events) if state else previous_events,
        'latency_s': round(time.perf_counter() - started, 4),
        'diagnostic': {'plan': asdict(state.plan), 'selected_tools': list(state.selected_tools),
                       'rearmed': state.rearmed, 'tool_rounds': state.tool_rounds,
                       'reads_executed': state.reads_executed, 'retrieval_refinements': state.retrieval_refinements,
                       'missing_evidence': list(state.missing_evidence),
                       'memory': asdict(state.memory), 'distilled_upto': state.distilled_upto,
                       'handoff_records': {k: asdict(v) for k,v in state.handoff_records.items()}}
                      if state and state.plan else {},
        **capture.since(position),
    }
    record['retrieved_source_ids'] = sorted({e['citation']['source_id'] for r in record['retrieval']
                                           for e in r['evidence'] if e['citation'].get('source_id')})
    record['retrieved_chunk_ids'] = sorted({e['citation']['chunk_id'] for r in record['retrieval'] for e in r['evidence']})
    # JSON is also the artifact boundary: dates become ISO strings and tuples
    # become lists consistently for both cached and output citations.
    record = json.loads(json.dumps(record, default=str))
    try:
        record['checks'] = check_turn(turn, record, replay=replay)
    except Exception as exc:
        # A checker fault fails this turn visibly instead of aborting the replay.
        record['checks'] = {'checker': {'passed': False, 'detail': f'{type(exc).__name__}: {exc}'}}
    record['failures'] = failures(record['checks'])
    record['passed'] = not record['failures']
    return record


async def run_session(app, session, *, capture, thread_id=None, verbose=True):
    """One ordered replay on one thread; accumulate evidence in graph checkpoints."""
    config = {'configurable': {'thread_id': thread_id or f"{session['id']}-{uuid.uuid4().hex[:8]}",
                               'caller': CallerContext(**session['caller'])},
              'recursion_limit': RECURSION_LIMIT}
    records, previous_events = [], 0
    for turn in session['turns']:
        record = await run_turn(app, turn, capture=capture, config=config, previous_events=previous_events)
        previous_events = record['total_handoff_events']
        records.append(record)
        if verbose:
            print_record(record)
    return records


def print_record(record):
    reads = ','.join(c['name'] for c in record['reads']) or '-'
    print(f"t{record['turn']:02d} {'PASS' if record['passed'] else 'FAIL'} intent={record['intent']} "
          f"reads={reads} retrieval={len(record['retrieval'])} handoffs={len(record['handoffs'])} "
          f"steps={len(record['nodes'])}/{RECURSION_LIMIT}")
    if record['failures']:
        for name, detail in record['failures'].items():
            print(f'  {name}: {detail}')
        diagnostic = deepcopy(record['diagnostic'])
        diagnostic.pop('handoff_records', None)
        print('  state: ' + json.dumps(diagnostic, default=str))
        print('  answer: ' + record['answer'])


async def run_permission_case(runtime):
    """A separate unrelated regular caller; denial before any dependency call."""
    position, backend_before = runtime.capture.position(), runtime.search.counts()
    try:
        result = await runtime.app.ainvoke(
            {'newest_message': "Show me Maya Cohen's onboarding evidence and offer letter."},
            {'configurable': {'thread_id': 'maya-unauthorized-regular',
                              'caller': CallerContext('E002','UG_REGULAR')}})
        response = response_to_dict(result['response'])
    except Exception as exc:
        checks = {'denied': {'passed': False, 'detail': f'{type(exc).__name__}: {exc}'}}
        return {'caller': {'employee_id':'E002','user_group':'UG_REGULAR'}, 'response': None,
                'calls': runtime.capture.since(position), 'checks': checks,
                'failures': failures(checks), 'passed': False}
    delta = runtime.capture.since(position)
    serialized = json.dumps(response).casefold()
    sensitive = ['maya', 'e001', 'customer success', '2026-08-01', 'offer_letter',
                 'employment/', 'chunk_id']
    checks = {
        'denied': {'passed': response['status'] == 'denied', 'detail': response['message']},
        'zero_dependency_calls': {'passed': not any(delta.values()), 'detail': str({k:len(v) for k,v in delta.items()})},
        'zero_backend_calls': {'passed': runtime.search.counts() == backend_before, 'detail': str(runtime.search.counts())},
        'no_sensitive_output': {'passed': not any(v in serialized for v in sensitive),
                                'detail': str([v for v in sensitive if v in serialized])},
        'no_evidence': {'passed': not response['evidence'] and not response['tasks']
                                  and response['role'] is None and response['start_date'] is None,
                       'detail': 'Denied response must contain no employee facts or evidence'},
    }
    return {'caller': {'employee_id':'E002','user_group':'UG_REGULAR'}, 'response':response,
            'calls':delta, 'backend_before':backend_before, 'backend_after':runtime.search.counts(),
            'checks':checks, 'failures':failures(checks), 'passed':not failures(checks)}


async def run_first_request_case(runtime, session, *, thread_id):
    """Sara's request alone: a complete, cited checklist in one turn."""
    config = {'configurable': {'thread_id': thread_id, 'caller': CallerContext(**session['caller'])},
              'recursion_limit': RECURSION_LIMIT}
    record = await run_turn(runtime.app, FIRST_REQUEST, capture=runtime.capture, config=config)
    checklist = record['checklist'] or {}
    retrieved = {s.rsplit('/', 1)[-1] for s in (e['citation']['source'] for r in record['retrieval'] for e in r['evidence'])}
    cited = {c['source'].rsplit('/', 1)[-1] for t in checklist.get('tasks', []) for c in t.get('evidence', [])}
    missing = [s for s in FIRST_REQUEST['required_sources'] if s not in retrieved]
    uncited = [alts for alts in FIRST_REQUEST['cited_sources'] if not set(alts.split('|')) & cited]
    capacity = [t for t in checklist.get('tasks', []) if t['status'] == 'blocked' and (
        '42' in t['description'] and any(c['source'].endswith(('webex_license_cleanup_memo.md',
            'check_software_subscription', 'webex_license_assignment.md')) for c in t['evidence']))]
    record['checks'].update({
        'identity_fields': {'passed': checklist.get('role') == 'Customer Success Manager'
                            and checklist.get('start_date') == '2026-08-01' and checklist.get('location') == 'Israel',
                            'detail': str({k: checklist.get(k) for k in ('role', 'start_date', 'location')})},
        'required_sources_retrieved': {'passed': not missing, 'detail': f'missing={missing}'},
        'sections_cited': {'passed': not uncited, 'detail': f'uncited={uncited}; cited={sorted(cited)}'},
        'blocker_cites_capacity': {'passed': bool(capacity), 'detail': 'Webex blocker must cite 42 seats vs limit'},
        'no_unresolved': {'passed': checklist.get('status') in ('ok', 'pending'),
                          'detail': str([t['description'] for t in checklist.get('tasks', []) if t['status'] == 'unresolved'])},
    })
    record['failures'] = failures(record['checks'])
    record['passed'] = not record['failures']
    return record


def static_read_only_check():
    return all(set(select_tools(ContextPlan(intent, requires_operational_reads=True), fallback=fallback)) <= READ_TOOLS
               for intent in [*LOADOUTS,'unknown'] for fallback in (False,True)) and 'create_access_request' not in READ_TOOLS


async def run_evaluation(runtime, session, *, thread_id=None, verbose=True):
    records = await run_session(runtime.app, session, capture=runtime.capture, thread_id=thread_id, verbose=verbose)
    thread = records[0]['thread_id']
    turn10 = next(t for t in session['turns'] if t['n'] == 10)
    original_ack = records[9]['checklist'].get('pending_access') if records[9]['checklist'] else None
    replay = await run_turn(runtime.app, turn10, capture=runtime.capture,
        config={'configurable': {'thread_id':thread, 'caller':CallerContext(**session['caller'])}, 'recursion_limit':RECURSION_LIMIT},
        previous_events=records[-1]['total_handoff_events'], replay=True)
    replay['checks']['same_acknowledgement'] = {
        'passed': bool(original_ack) and (replay['checklist'] or {}).get('pending_access') == original_ack,
        'detail': 'Replay must return the original pending request ID'}
    replay['checks']['one_port_call_total'] = {'passed':len(runtime.webex.calls) == 1, 'detail':f'fake calls={len(runtime.webex.calls)}'}
    replay['failures'] = failures(replay['checks'])
    replay['passed'] = not replay['failures']
    permission = await run_permission_case(runtime)
    first = await run_first_request_case(runtime, session, thread_id=f'{thread}-first-request')
    try:
        matrix, matrix_error = check_s2_matrix(), None
    except (AssertionError, ValueError, KeyError, OSError) as exc:
        matrix, matrix_error = [], f'{type(exc).__name__}: {exc}'
    session_checks = {
        'all_twelve_turns': len(records) == 12 and [r['turn'] for r in records] == list(range(1,13)),
        'one_stable_thread': len({r['thread_id'] for r in records}) == 1,
        'every_turn_passed': all(r['passed'] for r in records),
        'read_only_policy': static_read_only_check(),
        'one_handoff_at_turn10': sum(len(r['handoffs']) for r in records) == 1 and len(records[9]['handoffs']) == 1,
        'one_handoff_event_total': records[-1]['total_handoff_events'] == 1,
        'replay_idempotent': replay['passed'],
        'safe_permission_denial': permission['passed'],
        'first_request_checklist': first['passed'],
        'matrix_all_turns': not matrix_error and {r['turn'] for r in matrix} == set(range(1,13)),
    }
    report = {'session_id':session['id'], 'thread_id':thread, 'mode':'offline',
              'provenance':'Lesson 10 sequential runner adapted to Maya. Manual project S2 acceptance copy; original non-turn-10 expectations preserved. Search expectations map to injected retrieval.',
              'quality_scope':'Deterministic model test double plus actual local documents/records; does not certify live model or OpenSearch quality.',
              'records':records, 'replay':replay, 'permission':permission, 'first_request':first, 'matrix':matrix,
              'matrix_error':matrix_error,
              'session_checks':session_checks, 'passed':all(session_checks.values()),
              'failures_by_turn':[{'turn':r['turn'],'failures':r['failures'],'diagnostic':r['diagnostic']} for r in records if not r['passed']]}
    if verbose:
        print('Replay turn 10: ' + ('PASS' if replay['passed'] else 'FAIL'))
        if not replay['passed']: print_record(replay)
        print('Permission case: ' + ('PASS' if permission['passed'] else 'FAIL'))
        print('First request (own thread): ' + ('PASS' if first['passed'] else 'FAIL'))
        if not first['passed']: print_record(first)
        if not permission['passed']: print(json.dumps(permission, default=str))
        if matrix_error: print('Tool matrix: FAIL ' + matrix_error)
        print('Session: ' + ('PASS' if report['passed'] else 'FAIL'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--session', default='S2-onboarding-maya')
    parser.add_argument('--thread-id', default=None)
    parser.add_argument('--output', type=Path, default=RESULTS_DIR / 'latest-s2-offline.json')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        session = load_session(args.dataset, args.session)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'Cannot load session {args.session!r} from {args.dataset}: {type(exc).__name__}: {exc}')
    runtime = build_offline_runtime()
    report = asyncio.run(run_evaluation(runtime, session,
                                        thread_id=args.thread_id, verbose=not args.quiet))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    except OSError as exc:
        raise SystemExit(f'Cannot write report {args.output}: {exc}')
    print(f'Report: {args.output.resolve()}')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
