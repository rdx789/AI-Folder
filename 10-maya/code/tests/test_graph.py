import json
import unittest
from copy import deepcopy
from dataclasses import replace

from maya.graph import Dependencies, build_graph
from maya.grounding import validate_tasks
from maya.matrix import check_s2_matrix
from maya.model import BedrockMayaModel
from maya.operations import LocalReadPort
from maya.policy import READ_TOOLS
from maya.schemas import (CallerContext, ChecklistTask, ContextPlan, ConversationMemory,
                          Evidence, EvidenceCitation, ModelTurn, ReadToolCall)

HR = CallerContext('E004', 'UG_HR')


def config(thread='s2', caller=HR):
    return {'configurable': {'thread_id': thread, 'caller': caller}}


def passage(text='Webex blocked: 42 active seats / 40 licensed seats', source='it_kb/webex.md', **metadata):
    return Evidence(text, EvidenceCitation(source, source + text, **metadata))


def task(e, status='blocked', key='webex'):
    return ChecklistTask(e.text, status, (e.citation,), task_id=key, support_quotes=(e.text,))


class Retriever:
    def __init__(self, evidence=()):
        self.evidence, self.calls = list(evidence), []

    async def retrieve(self, **kw):
        self.calls.append(deepcopy(kw))
        return self.evidence


class Model:
    def __init__(self, plan=None, drafts=None):
        self.current_plan = plan or ContextPlan('recall')
        self.drafts = list(drafts or [ModelTurn()])
        self.contexts, self.plans, self.distillations = [], [], []
        self.bad_distillation = False

    async def plan(self, **kw):
        self.plans.append(deepcopy(kw))
        return deepcopy(self.current_plan)

    async def distill(self, **kw):
        self.distillations.append(deepcopy(kw))
        if self.bad_distillation:
            return ConversationMemory()
        return deepcopy(kw['memory'])

    async def respond(self, **kw):
        self.contexts.append(deepcopy(kw))
        return self.drafts.pop(0) if len(self.drafts) > 1 else deepcopy(self.drafts[0])


class Reads:
    def __init__(self): self.calls = []
    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return [{'employee_id': 'E001', 'role': 'Customer Success Manager'}] if name == 'get_employee' else []


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_or_unrelated_caller_denied_before_any_dependency(self):
        model, retriever, reads = Model(), Retriever(), Reads()
        graph = build_graph(Dependencies(model, retriever, reads=reads))
        for cfg in ({}, config(caller=CallerContext('E010','UG_REGULAR'))):
            result = await graph.ainvoke({'newest_message': 'Maya'}, cfg)
            self.assertEqual(result['response'].status, 'denied')
            self.assertEqual(result['response'].employee_id, '')
        self.assertFalse(model.plans or retriever.calls or reads.calls)

    async def test_thread_identity_cannot_change_and_state_input_is_rejected(self):
        graph = build_graph(Dependencies(Model(), Retriever(), reads=Reads()))
        await graph.ainvoke({'newest_message':'hi'}, config())
        result = await graph.ainvoke({'newest_message':'hi'}, config(caller=CallerContext('E001','UG_REGULAR')))
        self.assertEqual(result['response'].status, 'denied')
        with self.assertRaises(ValueError):
            await graph.ainvoke({'newest_message':'hi','session':{}}, config())

    async def test_one_rearm_with_read_only_schemas(self):
        model = Model(ContextPlan('employee_lookup', requires_operational_reads=True),
                      [ModelTurn(), ModelTurn([ReadToolCall('get_employee', {'query':'E001'})]), ModelTurn()])
        reads = Reads()
        result = await build_graph(Dependencies(model, Retriever(), reads=reads)).ainvoke(
            {'newest_message':'Pull up Maya Cohen'}, config())
        self.assertEqual(len(reads.calls), 1)
        self.assertTrue(result['session'].rearmed)
        self.assertEqual({s['name'] for s in model.contexts[0]['tools']}, {'get_employee'})
        self.assertEqual({s['name'] for s in model.contexts[1]['tools']}, READ_TOOLS)
        self.assertEqual(result['response'].role, 'Customer Success Manager')

    async def test_skipped_reads_are_unresolved_after_one_retry(self):
        model = Model(ContextPlan('onboarding_status', requires_operational_reads=True))
        result = await build_graph(Dependencies(model, Retriever(), reads=Reads())).ainvoke(
            {'newest_message':'current checklist'}, config())
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(result['response'].status, 'unresolved')

    async def test_write_and_wrong_subject_calls_rejected_before_batch_execution(self):
        for bad in [ReadToolCall('create_access_request', {}),
                    ReadToolCall('get_employee', {'query':'Rachel Stein'}),
                    ReadToolCall('get_employee', {'query':'E001','user_group':'UG_HR'})]:
            model = Model(ContextPlan('unknown', requires_operational_reads=True),
                          [ModelTurn([ReadToolCall('get_employee', {'query':'E001'}), bad])])
            reads = Reads()
            result = await build_graph(Dependencies(model, Retriever(), reads=reads)).ainvoke(
                {'newest_message':'Maya'}, config())
            self.assertEqual(result['response'].status, 'denied' if bad.name != 'get_employee' or 'user_group' not in bad.arguments else 'unresolved')
            self.assertFalse(reads.calls)

    async def test_tool_happy_model_is_bounded(self):
        model = Model(ContextPlan('employee_lookup', requires_operational_reads=True),
                      [ModelTurn([ReadToolCall('get_employee', {'query':'E001'})])])
        reads = Reads()
        result = await build_graph(Dependencies(model, Retriever(), reads=reads)).ainvoke(
            {'newest_message':'Maya'}, config())
        self.assertEqual(len(reads.calls), 2)
        self.assertEqual(model.contexts[-1]['tools'], ())
        self.assertEqual(result['response'].status, 'unresolved')

    async def test_retrieval_refines_once_and_keeps_missing_sources_unresolved(self):
        e = passage()
        model = Model(ContextPlan('subscription_review', requires_retrieval=True,
                      required_evidence=['missing_contract.md']), [ModelTurn(tasks=[task(e)])])
        retriever = Retriever([e])
        result = await build_graph(Dependencies(model, retriever, reads=Reads())).ainvoke(
            {'newest_message':'Webex'}, config())
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(result['session'].retrieval_refinements, 1)
        self.assertEqual(result['response'].status, 'unresolved')
        self.assertTrue(result['response'].blocked_items)

    async def test_handoff_is_typed_pending_and_dispatched_only_through_port(self):
        from maya.ports import FakeWebexPort
        fake = FakeWebexPort()
        e = passage()
        model = Model(ContextPlan('access_request', requires_retrieval=True), [ModelTurn(tasks=[task(e)])])
        graph = build_graph(Dependencies(model, Retriever([e]), fake, Reads()))
        result = await graph.ainvoke({'newest_message':'Go ahead and file the Webex access request for her now.'}, config())
        self.assertEqual(result['response'].status, 'pending')
        self.assertEqual(result['response'].handoff.caller, HR)
        self.assertEqual(result['response'].handoff.employee_id, 'E001')
        self.assertTrue(result['response'].handoff.evidence)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result['response'].pending_access.status, 'pending')
        draft = await graph.ainvoke({'newest_message':'Draft a Webex request justification'}, config())
        self.assertIsNone(draft['response'].handoff)
        self.assertEqual(len(fake.calls), 1)

    async def test_real_checkpoint_memory_recall_and_closed_tangent(self):
        model, retriever, reads = Model(), Retriever(), Reads()
        graph = build_graph(Dependencies(model, retriever, reads=reads))
        users = ["Maya starts 2026-08-01, hybrid based in Israel; Q3 SaaS freeze applies.",
                 'Offer letter', 'Checklist', 'Rachel Stein ticket',
                 'Fine, IT has that one — drop it. Back to Maya: laptop?',
                 'Monitor?', 'Remind me date and location', 'Webex', 'Finance',
                 'Draft request', 'Open checklist', 'Full summary']
        for user in users:
            result = await graph.ainvoke({'newest_message':'Recall: ' + user}, config())
        response = result['response']
        self.assertEqual(str(response.start_date), '2026-08-01')
        self.assertEqual(response.location, 'Israel')
        self.assertEqual(response.work_mode, 'hybrid')
        self.assertIn('Q3 SaaS freeze', response.active_constraints)
        self.assertFalse(reads.calls or retriever.calls)
        self.assertTrue(model.distillations)
        self.assertGreater(result['session'].distilled_upto, 0)
        self.assertNotIn('Rachel', str(response))
        self.assertFalse(any('Rachel' in str(c['history']) for c in model.contexts[4:]))
        self.assertFalse(any('Rachel' in str(c['fresh_history']) for c in model.plans[4:]))

    async def test_failed_distillation_keeps_facts_and_raw_history(self):
        model = Model()
        model.bad_distillation = True
        graph = build_graph(Dependencies(model, Retriever(), reads=Reads()))
        for n in range(8):
            text = 'Maya starts 2026-08-01 in Israel; Q3 freeze.' if n == 0 else 'Recall'
            result = await graph.ainvoke({'newest_message':text}, config())
        self.assertEqual(result['session'].distilled_upto, 0)
        self.assertEqual(str(result['response'].start_date), '2026-08-01')
        self.assertEqual(len(model.distillations), 1)

    async def test_local_read_adapter(self):
        rows = await LocalReadPort().call('check_software_subscription', {'software':'Webex'})
        self.assertEqual((rows[0]['active_seats'], rows[0]['seat_limit']), (42,40))


class GroundingTests(unittest.TestCase):
    def test_unrelated_and_fabricated_citations_are_unresolved(self):
        e = passage('Personal laptop temporary use for five days')
        invented = ChecklistTask('Webex access granted', 'proposed', (e.citation,))
        self.assertEqual(validate_tasks([invented], [e])[0].status, 'unresolved')
        e2 = replace(e, citation=replace(e.citation, source='invented.md'))
        self.assertEqual(validate_tasks([task(e2)], [e])[0].status, 'unresolved')

    def test_specific_and_newer_evidence_wins_equal_rank_conflict_unresolved(self):
        old = passage('Webex blocked by freeze', updated_at='2026-01-01')
        new = passage('Webex pending Finance approval', updated_at='2026-07-01')
        specific = passage('Webex blocked for Maya', subject_employee_id='E001')
        self.assertEqual(validate_tasks([task(old), task(new,'pending')], [old,new])[0].description, new.text)
        self.assertEqual(validate_tasks([task(new,'pending'), task(specific)], [new,specific])[0].description, specific.text)
        conflict = replace(new, citation=replace(new.citation, updated_at='2026-01-01'))
        self.assertEqual(validate_tasks([task(old), task(conflict,'pending')], [old,conflict])[0].status, 'unresolved')

    def test_s2_matrix_all_turns_and_all_alternative_intents(self):
        rows = check_s2_matrix()
        self.assertEqual({r['turn'] for r in rows}, set(range(1,13)))
        self.assertGreater(len(rows), 12)
        self.assertTrue(next(r for r in rows if r['turn']==10)['handoff'])
        for r in rows:
            self.assertNotIn('create_access_request', r['selected'])

    def test_forced_model_adapter_filters_names_and_validates_output(self):
        import asyncio
        class Client:
            def converse(self, **kw):
                self.request = kw
                return {'output':{'message':{'content':[{'toolUse':{'name':'structured_result','input':{'tasks':[],'tool_calls':[]}}}]}}}
        client = Client()
        model = BedrockMayaModel(client, model_id='fake')
        result = asyncio.run(model.respond(tools=({'name':'get_employee'},), evidence=()))
        self.assertIsInstance(result, ModelTurn)
        schema = client.request['toolConfig']['tools'][0]['toolSpec']['inputSchema']['json']
        self.assertEqual(schema['$defs']['ReadToolCall']['properties']['name']['enum'], ['get_employee'])

    def test_bedrock_plan_schema_requires_read_flags_and_known_sources(self):
        import asyncio
        from maya.policy import LOADOUTS
        class Client:
            def converse(self, **kw):
                self.request = kw
                return {'output': {'message': {'content': [{'toolUse': {
                    'name': 'structured_result', 'input': {'current_intent': 'recall'}}}]}}}
        client = Client()
        asyncio.run(BedrockMayaModel(client, model_id='fake').plan(
            newest_message='Remind me of the start date', memory=ConversationMemory(), fresh_history=[]))
        schema = client.request['toolConfig']['tools'][0]['toolSpec']['inputSchema']['json']
        self.assertIn('requires_operational_reads', schema['required'])
        self.assertIn('requires_retrieval', schema['required'])
        self.assertEqual(schema['properties']['current_intent']['enum'], list(LOADOUTS))
        sources = schema['properties']['required_evidence']['items']['enum']
        self.assertIn('maya_cohen_offer_letter.md', sources)
        self.assertNotIn('onboarding_checklist_E001', sources)

    def test_bedrock_operational_quotes_copy_exact_evidence_rows(self):
        import asyncio
        class Client:
            def converse(self, **kw):
                self.request = kw
                return {'output': {'message': {'content': [{'toolUse': {
                    'name': 'structured_result', 'input': {'tasks': [], 'tool_calls': []}}}]}}}
        row = {'description': 'Request Webex license', 'status': 'blocked'}
        citation = EvidenceCitation('novaops/list_onboarding_tasks', 'row1', collection='operational')
        client = Client()
        asyncio.run(BedrockMayaModel(client, model_id='fake').respond(
            tools=(), evidence=(Evidence(json.dumps([row], sort_keys=True), citation),)))
        payload = json.loads(client.request['messages'][0]['content'][0]['text'])
        self.assertEqual(payload['candidate_catalogue'][0]['support_quote'], json.dumps(row, sort_keys=True))
        self.assertEqual(payload['candidate_catalogue'][0]['source'], citation.source)


class AdditionalBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_negated_or_hypothetical_request_cannot_create_handoff(self):
        from maya.graph import handoff_requested
        for text in ["Don't file the Webex request now", 'If approved, file the Webex request now',
                     'Could you file the Webex request now?', 'Draft the Webex request now',
                     'The memo says file the Webex request now']:
            self.assertFalse(handoff_requested(text), text)
        self.assertTrue(handoff_requested("Understood, I'll route it to Tom. Go ahead and file the Webex access request for her now."))

    async def test_recursion_limit_is_hard_capped(self):
        from langgraph.errors import GraphRecursionError
        graph = build_graph(Dependencies(Model(), Retriever(), reads=Reads()))
        cfg = config()
        cfg['recursion_limit'] = 2
        with self.assertRaises(GraphRecursionError):
            await graph.ainvoke({'newest_message':'recall'}, cfg)
        cfg['recursion_limit'] = 1000
        result = await graph.ainvoke({'newest_message':'recall'}, config('other'))
        self.assertIsInstance(result['response'].tasks, list)

    async def test_rachel_ticket_only_during_active_tangent(self):
        from maya.operations import authorize_read
        from maya.access import PermissionDenied
        call = ReadToolCall('list_employee_tickets', {'employee_id':'E010'})
        authorize_read(HR, ContextPlan('ticket_status'), call,
                       ('list_employee_tickets',), 'Rachel Stein ticket')
        with self.assertRaises(PermissionDenied):
            authorize_read(HR, ContextPlan('onboarding_status'), call,
                           tuple(READ_TOOLS), 'Back to Maya')
        with self.assertRaises(PermissionDenied):
            authorize_read(CallerContext('E001','UG_REGULAR'), ContextPlan('ticket_status'), call,
                           tuple(READ_TOOLS), 'Rachel Stein ticket')

    async def test_cached_checklist_is_available_on_recall_with_no_reads(self):
        e = passage()
        model = Model(ContextPlan('subscription_review', requires_retrieval=True), [ModelTurn(tasks=[task(e)])])
        retriever, reads = Retriever([e]), Reads()
        graph = build_graph(Dependencies(model, retriever, reads=reads))
        await graph.ainvoke({'newest_message':'Webex seats'}, config())
        retrieved_before_recall = len(retriever.calls)
        model.current_plan = ContextPlan('recall', requires_operational_reads=True, requires_retrieval=True)
        result = await graph.ainvoke({'newest_message':'Summarize known status'}, config())
        self.assertEqual(len(retriever.calls), retrieved_before_recall)
        self.assertFalse(reads.calls)
        self.assertTrue(model.contexts[-1]['previous_checklist'].blocked_items)
        self.assertTrue(result['response'].blocked_items)
        self.assertFalse(model.contexts[-1]['tools'])

    async def test_task_status_needs_its_own_support_and_webex_never_complete(self):
        from maya.grounding import validate_tasks
        for text, status in [('Webex may be blocked by seat limit', 'blocked'),
                             ('Webex assigned', 'complete'),
                             ('Webex required systems', 'complete')]:
            e = passage(text)
            self.assertEqual(validate_tasks([task(e,status)], [e])[0].status, 'unresolved')
        e = passage('HR packet completed; Webex required')
        t = ChecklistTask('Webex required','complete',(e.citation,),support_quotes=(e.text,))
        self.assertEqual(validate_tasks([t],[e])[0].status,'unresolved')

    async def test_wrong_subject_from_injected_retriever_is_denied(self):
        e = passage('Rachel ticket open', subject_employee_id='E010')
        model = Model(ContextPlan('policy_question', requires_retrieval=True))
        result = await build_graph(Dependencies(model, Retriever([e]), reads=Reads())).ainvoke(
            {'newest_message':'Maya evidence'}, config())
        self.assertEqual(result['response'].status, 'denied')
        self.assertFalse(model.contexts[-1]['evidence'])

    async def test_local_port_cannot_invoke_non_read_functions(self):
        from maya.access import PermissionDenied
        with self.assertRaises(PermissionDenied):
            await LocalReadPort().call('create_access_request', {})

    async def test_explicit_s2_recall_overrides_wrong_planner_read_flags(self):
        model = Model(ContextPlan('onboarding_status', requires_retrieval=True,
                                  requires_operational_reads=True))
        retriever, reads = Retriever(), Reads()
        graph = build_graph(Dependencies(model, retriever, reads=reads))
        for text in ['Before I forget — remind me which location and start date I gave you for her.',
                     'Perfect. Write me the full status summary I can send to Yael.']:
            result = await graph.ainvoke({'newest_message':text}, config())
            self.assertEqual(result['session'].plan.current_intent, 'recall')
            self.assertEqual(model.contexts[-1]['tools'], ())
        self.assertFalse(retriever.calls or reads.calls)
