"""Milestone 4: real checkpoint replay with a deliberately non-deduplicating fake."""
import asyncio
import json
import unittest
from dataclasses import replace

from maya.graph import Dependencies, build_graph, WEBEX_BUSINESS_REASON
from maya.ingestion import load_chunks
from maya.paths import CODE_DIR, DATA_PATH, PROJECT_DIR
from maya.ports import FakeWebexPort, WebexDispatchGuard, webex_request_key
from maya.policy import READ_TOOLS
from maya.schemas import CallerContext, ChecklistTask, ContextPlan, Evidence, EvidenceCitation, ModelTurn, ReadToolCall, WebexHandoff
from test_graph import Model, Retriever, Reads, config

SESSION = json.loads((CODE_DIR / 'evals/data/sessions.json').read_text())['sessions'][0]
TURN10 = next(t for t in SESSION['turns'] if t['n'] == 10)
CALLER = CallerContext(**SESSION['caller'])


def webex_evidence():
    record = next(r for r in load_chunks(DATA_PATH) if r['source'] == 'memos/webex_license_cleanup_memo.md')
    return Evidence(record['text'], EvidenceCitation(
        source=record['source'], source_id=record['source_id'], chunk_id=record['chunk_id'],
        collection=record['collection'], audience=tuple(record['audience']),
        sensitivity=record['sensitivity'], updated_at=record['updated_at']))


def grounded_task(evidence):
    return ChecklistTask(evidence.text, 'blocked', (evidence.citation,),
                         task_id='webex', support_quotes=(evidence.text,))


def fixture(*, port=None, checkpointer=None):
    evidence = webex_evidence()
    model = Model(ContextPlan('access_request', requires_retrieval=True),
                  [ModelTurn(tasks=[grounded_task(evidence)])])
    retriever, reads = Retriever([evidence]), Reads()
    fake = port if port is not None else FakeWebexPort()
    graph = build_graph(Dependencies(model, retriever, fake, reads), checkpointer=checkpointer)
    return graph, fake, model, retriever, reads


class HandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn10_and_exact_replay_one_call_one_event_stable_pending_result(self):
        graph, fake, model, retriever, reads = fixture()
        first = await graph.ainvoke({'newest_message': TURN10['user']}, config(caller=CALLER))
        second = await graph.ainvoke({'newest_message': TURN10['user']}, config(caller=CALLER))
        third = await graph.ainvoke({'newest_message': TURN10['user']}, config(caller=CALLER))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(third['session'].handoff_events), 1)
        expected = TURN10['expected_handoffs'][0]
        handoff = fake.calls[0]
        self.assertEqual(handoff.kind, expected['kind'])
        self.assertEqual(handoff.employee_id, expected['employee_id'])
        self.assertEqual(handoff.system, expected['system'])
        self.assertEqual(handoff.caller, CALLER)
        self.assertTrue(handoff.business_reason)
        self.assertTrue(handoff.evidence)
        self.assertEqual(handoff.idempotency_key, webex_request_key(
            thread_id='s2', caller=CALLER, business_reason=WEBEX_BUSINESS_REASON))
        for result in (first, second, third):
            response = result['response']
            self.assertEqual(response.pending_access.status, expected['status'])
            self.assertEqual(response.status, 'pending')
            self.assertEqual(response.pending_access, first['response'].pending_access)
            self.assertIn(response.pending_access.request_id, response.message)
            self.assertIn('pending approval', response.message)
            self.assertTrue(response.blocked_items)
            self.assertNotIn('access granted', response.message.lower())
        self.assertFalse(reads.calls)
        for context in model.contexts:
            self.assertTrue({t['name'] for t in context['tools']} <= READ_TOOLS)
            self.assertNotIn('create_access_request', {t['name'] for t in context['tools']})

    async def test_same_semantic_request_different_wording_reuses_key(self):
        graph, fake, *_ = fixture()
        first = await graph.ainvoke({'newest_message': TURN10['user']}, config())
        second = await graph.ainvoke({'newest_message': 'Please submit the Webex request now.'}, config())
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(first['response'].pending_access, second['response'].pending_access)

    async def test_simultaneous_replays_serialize_thread_and_call_once(self):
        class SlowFake(FakeWebexPort):
            async def request_access(self, handoff):
                await asyncio.sleep(0)
                return await super().request_access(handoff)
        graph, fake, *_ = fixture(port=SlowFake())
        results = await asyncio.gather(*(graph.ainvoke({'newest_message': TURN10['user']}, config()) for _ in range(3)))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len({r['response'].pending_access.request_id for r in results}), 1)
        self.assertEqual(len(results[-1]['session'].handoff_events), 1)

    async def test_rebuild_graph_same_checkpoint_does_not_call_a_new_port(self):
        graph, fake, *_ = fixture()
        first = await graph.ainvoke({'newest_message': TURN10['user']}, config())
        # New graph, dispatcher and port: deduplication must come from the checkpoint.
        rebuilt, second_fake, *_ = fixture(checkpointer=graph.compiled.checkpointer)
        replay = await rebuilt.ainvoke({'newest_message': TURN10['user']}, config())
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(second_fake.calls), 0)
        self.assertEqual(replay['response'].pending_access, first['response'].pending_access)
        self.assertEqual(len(replay['session'].handoff_events), 1)

    async def test_different_threads_have_different_request_keys(self):
        graph, fake, *_ = fixture()
        a = await graph.ainvoke({'newest_message': TURN10['user']}, config('one'))
        b = await graph.ainvoke({'newest_message': TURN10['user']}, config('two'))
        self.assertEqual(len(fake.calls), 2)
        self.assertNotEqual(a['response'].handoff.idempotency_key, b['response'].handoff.idempotency_key)

    async def test_pending_result_survives_recall_and_distillation_without_redispatch(self):
        graph, fake, model, retriever, reads = fixture()
        result = await graph.ainvoke({'newest_message': SESSION['turns'][0]['user']}, config())
        first = await graph.ainvoke({'newest_message': TURN10['user']}, config())
        model.current_plan = ContextPlan('recall')
        for _ in range(8):
            result = await graph.ainvoke({'newest_message': 'Perfect. Write me the full status summary I can send to Yael.'}, config())
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result['response'].pending_access, first['response'].pending_access)
        self.assertEqual(str(result['response'].start_date), '2026-08-01')
        self.assertEqual(result['response'].location, 'Israel')
        self.assertIn('Q3 SaaS freeze', result['response'].active_constraints)
        self.assertTrue(model.distillations)
        self.assertEqual(len(result['session'].handoff_records), 1)
        self.assertEqual(len(result['session'].handoff_events), 1)

    async def test_missing_port_never_claims_an_acknowledged_pending_handoff(self):
        graph, _, model, retriever, reads = fixture()
        graph = build_graph(Dependencies(model, retriever, reads=reads))
        result = await graph.ainvoke({'newest_message': TURN10['user']}, config())
        self.assertEqual(result['response'].status, 'unresolved')
        self.assertIsNone(result['response'].pending_access)
        self.assertFalse(result['session'].handoff_records or result['session'].handoff_events)

    async def test_unauthorized_caller_and_thread_owner_make_no_port_call(self):
        graph, fake, *_ = fixture()
        denied = await graph.ainvoke({'newest_message': TURN10['user']}, config(caller=CallerContext('E010','UG_REGULAR')))
        self.assertEqual(denied['response'].status, 'denied')
        self.assertFalse(fake.calls)
        await graph.ainvoke({'newest_message': TURN10['user']}, config())
        denied = await graph.ainvoke({'newest_message': TURN10['user']}, config(caller=CallerContext('E001','UG_REGULAR')))
        self.assertEqual(denied['response'].status, 'denied')
        self.assertEqual(len(fake.calls), 1)

    async def test_draft_negation_and_missing_evidence_make_no_port_call(self):
        for text in ['Draft the Webex request', "Don't file the Webex request now", 'If approved, file the Webex request now']:
            graph, fake, *_ = fixture()
            await graph.ainvoke({'newest_message': text}, config())
            self.assertFalse(fake.calls)
        graph, fake, model, retriever, *_ = fixture()
        retriever.evidence = []
        result = await graph.ainvoke({'newest_message': TURN10['user']}, config())
        self.assertEqual(result['response'].status, 'unresolved')
        self.assertFalse(fake.calls)

    async def test_lost_acknowledgement_replay_is_unresolved_not_a_second_call(self):
        class LostAck(FakeWebexPort):
            async def request_access(self, handoff):
                await super().request_access(handoff)
                raise TimeoutError('Internal workflow detail must not appear in output')
        graph, fake, *_ = fixture(port=LostAck())
        for _ in range(2):
            result = await graph.ainvoke({'newest_message': TURN10['user']}, config())
            self.assertEqual(result['response'].status, 'unresolved')
            self.assertIsNone(result['response'].pending_access)
            self.assertNotIn('Internal workflow detail', str(result['response']))
        self.assertEqual(len(fake.calls), 1)
        self.assertFalse(result['session'].handoff_events)

    async def test_invalid_or_granted_acknowledgement_is_rejected_and_not_retried(self):
        class Granted(FakeWebexPort):
            async def request_access(self, handoff):
                result = await super().request_access(handoff)
                object.__setattr__(result, 'status', 'granted')
                return result
        graph, fake, *_ = fixture(port=Granted())
        for _ in range(2):
            result = await graph.ainvoke({'newest_message': TURN10['user']}, config())
            self.assertEqual(result['response'].status, 'unresolved')
            self.assertIsNone(result['response'].pending_access)
        self.assertEqual(len(fake.calls), 1)

    async def test_reservation_is_checkpointed_before_entering_port(self):
        graph = None
        class CheckReservation(FakeWebexPort):
            async def request_access(self, handoff):
                snapshot = await graph.compiled.aget_state(config())
                record = snapshot.values['session'].handoff_records[handoff.idempotency_key]
                if record.status != 'dispatching' or record.result is not None:
                    raise AssertionError('Attempt must be recorded before dispatch')
                return await super().request_access(handoff)
        graph, fake, *_ = fixture(port=CheckReservation())
        result = await graph.ainvoke({'newest_message': TURN10['user']}, config())
        self.assertEqual(result['response'].status, 'pending')
        self.assertEqual(len(fake.calls), 1)

    async def test_crash_after_reservation_never_resends_on_replay(self):
        from langgraph.errors import NodeCancelledError
        class CrashAfterSend(FakeWebexPort):
            async def request_access(self, handoff):
                await super().request_access(handoff)
                raise asyncio.CancelledError()
        graph, fake, *_ = fixture(port=CrashAfterSend())
        with self.assertRaises(NodeCancelledError):
            await graph.ainvoke({'newest_message': TURN10['user']}, config())
        rebuilt, next_fake, *_ = fixture(checkpointer=graph.compiled.checkpointer)
        replay = await rebuilt.ainvoke({'newest_message': TURN10['user']}, config())
        self.assertEqual(len(fake.calls), 1)
        self.assertFalse(next_fake.calls)
        self.assertEqual(replay['response'].status, 'unresolved')
        self.assertIsNone(replay['response'].pending_access)

    async def test_longest_rearm_two_read_round_path_still_fits_recursion_bound(self):
        graph, fake, model, *_ = fixture()
        evidence = webex_evidence()
        model.current_plan = ContextPlan('access_request', requires_operational_reads=True, requires_retrieval=True)
        model.drafts = [ModelTurn(), ModelTurn([ReadToolCall('get_employee', {'query':'E001'})]),
                        ModelTurn([ReadToolCall('check_software_subscription', {'software':'Webex'})]),
                        ModelTurn(tasks=[grounded_task(evidence)])]
        result = await graph.ainvoke({'newest_message': TURN10['user']}, config())
        self.assertTrue(result['session'].rearmed)
        self.assertEqual(result['session'].tool_rounds, 2)
        self.assertEqual(result['response'].status, 'pending')
        self.assertEqual(len(fake.calls), 1)

    async def test_fake_records_every_invocation_so_graph_deduplication_is_observable(self):
        fake = FakeWebexPort()
        handoff = WebexHandoff(CALLER, 'E001', WEBEX_BUSINESS_REASON,
            webex_request_key(thread_id='s2',caller=CALLER,business_reason=WEBEX_BUSINESS_REASON))
        first = await fake.request_access(handoff)
        second = await fake.request_access(handoff)
        self.assertEqual(first, second)
        self.assertEqual(len(fake.calls), 2)

    async def test_runtime_guard_prevents_duplicate_invocations_and_key_collision(self):
        fake = FakeWebexPort()
        guard = WebexDispatchGuard(fake)
        handoff = WebexHandoff(CALLER, 'E001', WEBEX_BUSINESS_REASON,
            webex_request_key(thread_id='s2',caller=CALLER,business_reason=WEBEX_BUSINESS_REASON))
        results = await asyncio.gather(*(guard.request_access(handoff) for _ in range(3)))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(results[0], results[-1])
        with self.assertRaises(ValueError):
            await guard.request_access(replace(handoff, business_reason='A different business request'))
        self.assertEqual(len(fake.calls), 1)


class SessionContractTests(unittest.TestCase):
    def test_original_turns_are_preserved_except_turn10_and_caller_fields(self):
        original_path = PROJECT_DIR.parent / 'lesson-10-context-engineering/code/evals/data/sessions.json'
        if not original_path.is_file():
            self.skipTest('Read-only classroom reference unavailable')
        original = next(s for s in json.loads(original_path.read_text())['sessions'] if s['id'] == SESSION['id'])
        self.assertEqual({k:v for k,v in SESSION.items() if k not in ('caller','turns')},
                         {k:v for k,v in original.items() if k != 'turns'})
        for expected, actual in zip(original['turns'], SESSION['turns']):
            if expected['n'] != 10:
                self.assertEqual(expected, actual)
            else:
                for key in ('n','user','expected_intent','kind','optional_tools'):
                    self.assertEqual(expected[key], actual[key])
        self.assertEqual(len(SESSION['turns']), 12)
        self.assertEqual(SESSION['caller'], {'employee_id':'E004','user_group':'UG_HR'})
        self.assertIn('create_access_request', TURN10['forbidden_tools'])
        self.assertNotIn('create_access_request', TURN10['expected_tools'])
        self.assertEqual(TURN10['expected_handoffs'][0]['count'], 1)
