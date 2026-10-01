"""Deterministic grading adapted from Lesson 10 evalrules, without model judges."""
import re

from ..graph import RECURSION_LIMIT
from ..policy import READ_TOOLS

SEARCH_COLLECTIONS = {'search_hr_documents': {'employment','memos','contracts'},
                      'search_knowledge_base': {'it_kb','policies'}}


def visible_status(checklist):
    """Grade displayed facts, never hash IDs, user input or retrieved passages."""
    if not checklist:
        return ''
    fields = [checklist.get(k) for k in ('full_name','employee_id','role','start_date','location','work_mode','message')]
    fields += checklist.get('active_constraints', [])
    for task in checklist.get('tasks', []) + checklist.get('related_status', []):
        fields += [task.get('description'), task.get('status'), task.get('reason')]
    if checklist.get('pending_access'):
        fields += list(checklist['pending_access'].values())
    return '\n'.join(str(v) for v in fields if v is not None)


def check_turn(turn, record, *, replay=False):
    checks = {}
    def check(name, passed, detail=''):
        checks[name] = {'passed': bool(passed), 'detail': detail}

    calls = record.get('reads', [])
    called = {c['name'] for c in calls}
    retrieval = record.get('retrieval', [])
    collections = {e['citation']['collection'] for r in retrieval for e in r['evidence']}
    expected = turn.get('expected_tools', [])
    missing = [name for name in expected if (
        not (retrieval and bool(collections & SEARCH_COLLECTIONS[name])) if name in SEARCH_COLLECTIONS else name not in called)]
    check('expected_reads', not missing, f'missing={missing}; called={sorted(called)}; retrieved_collections={sorted(collections)}')
    forbidden = [name for name in turn.get('forbidden_tools', [])
                 if name in called or (name in SEARCH_COLLECTIONS and retrieval)]
    check('forbidden_reads', not forbidden, f'forbidden calls={forbidden}')
    accessed = bool(calls or retrieval)
    required = turn.get('should_use_tools')
    check('read_necessity', required is None or accessed == required,
          f'expected reads/retrieval={required}; actual={accessed}')
    exposed = {s['name'] for m in record.get('models', []) for s in m['schemas']}
    requested = {c['name'] for m in record.get('models', []) for c in m['requested_calls']}
    check('read_only_catalogue', (exposed | requested | called) <= READ_TOOLS,
          f'non-read names={sorted((exposed | requested | called) - READ_TOOLS)}')
    check('calls_reachable', called <= exposed, f'called without exposure={sorted(called - exposed)}')
    for rule in turn.get('arg_checks', []):
        values = [c['arguments'].get(rule['arg']) for c in calls if c['name'] == rule['tool']]
        passed = bool(values) and any(
            ('equals' not in rule or v == rule['equals']) and
            ('contains' not in rule or rule['contains'].casefold() in str(v).casefold()) for v in values)
        check('argument:' + rule['tool'] + ':' + rule['arg'], passed, f'expected={rule}; actual={values}')
    output = visible_status(record.get('checklist')).casefold()
    for fact in turn.get('required_facts', []):
        check('fact:' + fact, any(alt.casefold() in output for alt in fact.split('|')), f'missing displayed fact={fact}')
    checklist = record.get('checklist') or {}
    if turn['n'] >= 5:
        scope = output + ' ' + ' '.join(str(c.get('subject_employee_id') or '') for c in checklist.get('evidence', []))
        leaks = [word for word in ('rachel', 't001', 'e010') if word in scope.casefold()]
        check('closed_tangent', not leaks, f'closed tangent references={leaks}')
    cached = {e['citation']['chunk_id']: e for e in record.get('cached_evidence', [])}
    citation_errors = []
    tasks = checklist.get('tasks', []) + checklist.get('related_status', [])
    for task in tasks:
        if task['status'] == 'unresolved':
            if not task.get('reason'):
                citation_errors.append(f"{task['task_id']}: unresolved without reason")
            continue
        citations = task.get('evidence', [])
        if not citations:
            citation_errors.append(f"{task['task_id']}: resolved task without citation")
            continue
        passages = []
        for citation in citations:
            item = cached.get(citation['chunk_id'])
            if not citation['source'] or not item or item['citation'] != citation:
                citation_errors.append(f"{task['task_id']}: unknown or mutated citation")
            else:
                passages.append(item['text'])
        normalize = lambda text: ' '.join(text.casefold().split())
        quotes = task.get('support_quotes') or [task['description']]
        if not all(any(normalize(q) in normalize(p) for p in passages) for q in quotes):
            citation_errors.append(f"{task['task_id']}: unsupported quote")
        if not any(normalize(task['description']) in normalize(q) for q in quotes):
            citation_errors.append(f"{task['task_id']}: unsupported task description")
    check('task_citations', not citation_errors, '; '.join(citation_errors))
    check('structured_checklist', bool(checklist) and checklist.get('employee_id') == 'E001'
          and isinstance(checklist.get('tasks'), list), 'Expected typed Maya checklist/status')
    check('runtime', not record.get('error') and not record.get('errors'),
          f"error={record.get('error')}; graph errors={record.get('errors')}")
    check('recursion_bound', bool(record.get('nodes')) and len(record['nodes']) <= RECURSION_LIMIT
          and not (record.get('error') or '').startswith('GraphRecursionError'),
          f"steps={len(record.get('nodes', []))}; limit={RECURSION_LIMIT}")
    expected_handoffs = turn.get('expected_handoffs', [])
    count = 0 if replay else sum(h.get('count', 1) for h in expected_handoffs)
    check('handoff_count', len(record.get('handoffs', [])) == count,
          f"expected new calls={count}; actual={len(record.get('handoffs', []))}")
    check('handoff_events', len(record.get('handoff_events', [])) == count,
          f"expected new events={count}; actual={len(record.get('handoff_events', []))}")
    for expected_handoff in expected_handoffs:
        actual = checklist.get('handoff') or {}
        acknowledgement = checklist.get('pending_access') or {}
        fields_match = all(actual.get(k) == v for k, v in expected_handoff.items() if k not in ('status','count'))
        caller = turn.get('handoff_caller', record.get('caller'))
        check('handoff_fields', fields_match and actual.get('caller') == caller and bool(actual.get('business_reason'))
              and bool(actual.get('idempotency_key')), f'expected={expected_handoff}; actual={actual}')
        check('pending_acknowledgement', acknowledgement.get('status') == expected_handoff.get('status','pending')
              and bool(acknowledgement.get('request_id')), f'acknowledgement={acknowledgement}')
    if turn['n'] >= 8:
        webex_tasks = [t for t in tasks if 'webex' in t.get('description','').casefold()]
        check('webex_not_granted', not any(t['status'] == 'complete' for t in webex_tasks)
              and not re.search(r'\b(?:access (?:is |was )?granted|license (?:is |was )?granted)\b', output),
              'Webex access must remain blocked/pending')
    if turn['n'] == 12:
        check('final_pending_request', (checklist.get('pending_access') or {}).get('status') == 'pending', 'Final summary lost pending acknowledgement')
        check('final_constraints', checklist.get('start_date') == '2026-08-01' and checklist.get('location') == 'Israel'
              and checklist.get('work_mode') == 'hybrid' and any('Q3' in c for c in checklist.get('active_constraints', [])),
              'Final summary must retain date, Israel/hybrid and Q3 freeze')
        check('final_finance', 'finance' in output and ('7,500' in output or '7500' in output), 'Final summary must retain current Finance threshold')
    return checks


def failures(checks):
    return {name: result['detail'] for name, result in checks.items() if not result['passed']}
