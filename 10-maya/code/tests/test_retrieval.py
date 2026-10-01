"""Offline query-contract tests over the real local NovaOps Markdown corpus."""
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from maya.access import PermissionDenied
from maya.ingestion import index_body, ingest, load_chunks
from maya.reranker import BedrockReranker
from maya.paths import DATA_PATH
from maya.retrieval import OpenSearchEvidenceRetriever, retrieve_evidence_node, source_filter
from maya.schemas import CallerContext, ContextPlan

ROOT = DATA_PATH
HR = CallerContext('E004', 'UG_HR')
REGULAR = CallerContext('E002', 'UG_REGULAR')
MAYA = CallerContext('E001', 'UG_REGULAR')


def matches(query, record):
    """Independent tiny evaluator of OpenSearch term/terms/exists/bool semantics."""
    if 'term' in query:
        field, value = next(iter(query['term'].items()))
        actual = record.get(field)
        return value in actual if isinstance(actual, list) else actual == value
    if 'terms' in query:
        field, values = next(iter(query['terms'].items()))
        actual = record.get(field)
        return bool(set(actual) & set(values)) if isinstance(actual, list) else actual in values
    if 'exists' in query:
        return record.get(query['exists']['field']) is not None
    b = query['bool']
    return (all(matches(x, record) for x in b.get('must', []))
            and not any(matches(x, record) for x in b.get('must_not', []))
            and sum(matches(x, record) for x in b.get('should', [])) >= b.get('minimum_should_match', 0))


class SearchFake:
    def __init__(self, records):
        self.records = records
        self.calls = []
        self.filtered = []

    def search(self, *, index, body):
        self.calls.append(body)
        knn = body['query']['knn']['vector']
        # Filter before the candidate cut; forbidden documents never become candidates.
        eligible = [r for r in self.records if matches(knn['filter'], r)]
        self.filtered = eligible
        return {'hits': {'hits': [{'_source': r} for r in eligible[:knn['k']]]}}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.records = load_chunks(ROOT)
        self.search = SearchFake(self.records)
        self.embed = Mock(return_value=[1.0, 0.0])
        self.rank_calls = []

        def rank(query, candidates):
            self.rank_calls.append(candidates)
            return sorted(candidates, key=lambda r: (r['source'].split('/')[-1] not in query, r['source']))
        self.rank = rank
        self.adapter = OpenSearchEvidenceRetriever(self.search, index='test', embed=self.embed,
                                                   rerank=rank, candidate_count=50, evidence_count=4)

    def node(self, plan, caller=HR, retriever=None):
        return asyncio.run(retrieve_evidence_node(plan, {'configurable': {'caller': caller}},
                                                 retriever or self.adapter))

    def plan(self, filename='maya_cohen_offer_letter.md', subject='E001'):
        return ContextPlan('onboarding', subject_employee_id=subject,
                           retrieval_query=filename, required_evidence=[filename], requires_retrieval=True)

    def test_hr_can_cite_all_required_sources_and_blocker(self):
        names = ['maya_cohen_offer_letter.md', 'maya_cohen_onboarding_summary.md',
                 'onboarding_policy.md', 'equipment_policy.md', 'access_management_policy.md',
                 'salesforce_access_request.md', 'webex_license_assignment.md', 'laptop_provisioning.md',
                 'webex_license_cleanup_memo.md']
        for name in names:
            with self.subTest(source=name):
                result = self.node(self.plan(name))
                self.assertEqual(result.status, 'ok')
                evidence = next(e for e in result.evidence if e.citation.source.endswith(name))
                self.assertTrue(evidence.citation.source_id)
                self.assertTrue(evidence.citation.chunk_id)
                self.assertTrue(evidence.citation.collection)
        offer = self.node(self.plan()).evidence[0]
        self.assertIn('Customer Success Manager', offer.text)
        self.assertIn('2026-08-01', offer.text)
        blocker = self.node(self.plan('webex_license_cleanup_memo.md')).evidence[0]
        self.assertIn('42 active seats', blocker.text)
        self.assertIn('40-seat', blocker.text)

    def test_unrelated_employee_denied_without_any_inference_calls(self):
        result = self.node(self.plan(), REGULAR)
        self.assertEqual(result.status, 'denied')
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.missing, ())
        serialized = json.dumps(asdict(result))
        for forbidden in ('Maya', 'E001', 'Customer Success', '2026-08-01', 'offer_letter', 'chunk_id'):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(self.search.calls, [])
        self.embed.assert_not_called()
        self.assertEqual(self.rank_calls, [])

    def test_direct_adapter_cannot_bypass_subject_denial(self):
        with self.assertRaises(PermissionDenied):
            asyncio.run(self.adapter.retrieve(caller=REGULAR, plan=self.plan()))
        self.assertFalse(self.search.calls)

    def test_regular_employee_own_permitted_document(self):
        result = self.node(self.plan(), MAYA)
        self.assertEqual(result.status, 'ok')
        self.assertTrue(any(e.citation.subject_employee_id == 'E001' for e in result.evidence))

    def test_regular_own_scope_cannot_infer_coworker_even_from_query(self):
        result = self.node(self.plan(subject='E002'), REGULAR)
        self.assertEqual(result.status, 'unresolved')
        self.assertEqual(result.refinements, 1)
        for candidates in self.rank_calls:
            self.assertTrue(all(r.get('subject_employee_id') is None for r in candidates))
            self.assertTrue(all('Maya' not in r['text'] for r in candidates))
        self.assertTrue(all(e.citation.collection != 'employment' for e in result.evidence))

    def test_hard_filter_inside_knn_and_wide_then_narrow(self):
        result = self.node(self.plan())
        body = self.search.calls[0]
        self.assertNotIn('post_filter', body)
        knn = body['query']['knn']['vector']
        self.assertEqual(knn['k'], 50)
        self.assertIn('UG_HR', json.dumps(knn['filter']))
        self.assertGreater(len(self.rank_calls[0]), len(result.evidence))
        self.assertLessEqual(len(result.evidence), 4)
        self.assertFalse(any(r.get('subject_employee_id') == 'E010' for r in self.rank_calls[0]))

    def test_missing_runtime_identity_and_model_style_dict_are_denied(self):
        for config in ({}, {'configurable': {'caller': {'employee_id': 'E004', 'user_group': 'UG_HR'}}}):
            result = asyncio.run(retrieve_evidence_node(self.plan(), config, self.adapter))
            self.assertEqual(result.status, 'denied')
        self.assertFalse(self.search.calls)
        self.assertNotIn('user_group', ContextPlan.__dataclass_fields__)

    def test_unknown_groups_fail_closed(self):
        for group in ('manager', 'UG_ADMIN', '', 'ug_hr'):
            self.assertEqual(self.node(self.plan(), CallerContext('E004', group)).status, 'denied')
        self.embed.assert_not_called()

    def test_backend_acl_violation_never_reaches_reranker(self):
        forbidden = next(r for r in self.records if 'maya_cohen_offer' in r['source'])
        self.search.search = Mock(return_value={'hits': {'hits': [{'_source': forbidden}]}})
        result = self.node(self.plan('onboarding_policy.md', subject='E002'), REGULAR)
        self.assertEqual(result.status, 'denied')
        self.assertEqual(result.evidence, ())
        self.assertEqual(self.rank_calls, [])

    def test_missing_evidence_refines_once_with_same_acl_and_never_more(self):
        result = self.node(self.plan('missing_contract.md'))
        self.assertEqual(result.status, 'unresolved')
        self.assertEqual(result.refinements, 1)
        self.assertEqual(result.missing, ('missing_contract.md',))
        self.assertEqual(len(self.search.calls), 2)
        first = self.search.calls[0]['query']['knn']['vector']['filter']['bool']['must']
        refined = self.search.calls[1]['query']['knn']['vector']['filter']['bool']['must']
        self.assertEqual(refined[:len(first)], first)  # same caller ACL and subject scope
        self.assertEqual(refined[len(first):], [source_filter(('missing_contract.md',))])
        self.assertIn('Required sources', self.embed.call_args_list[1].args[0])

    def test_success_and_no_requirements_do_not_refine(self):
        self.node(self.plan())
        self.assertEqual(len(self.search.calls), 1)
        self.search.calls.clear()
        self.node(replace(self.plan(), required_evidence=[]))
        self.assertEqual(len(self.search.calls), 1)

    def test_recall_has_no_calls(self):
        self.assertEqual(self.node(ContextPlan('recall')).status, 'skipped')
        self.embed.assert_not_called()

    def test_one_refinement_can_fill_missing_source(self):
        source = self.node(self.plan()).evidence[0]
        fake = Mock()
        from unittest.mock import AsyncMock
        fake.retrieve = AsyncMock(side_effect=[[], [source]])
        result = self.node(self.plan(), retriever=fake)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.refinements, 1)
        self.assertEqual(fake.retrieve.await_count, 2)
        self.assertEqual(fake.retrieve.call_args.kwargs['caller'], HR)

    def test_refinement_fetches_named_source_that_ranks_poorly(self):
        # Live S2 turn 9: the freeze memo never reached the top 4 by relevance.
        def bury(query, candidates):
            return sorted(candidates, key=lambda r: ('saas_renewal_freeze_q3' in r['source'], r['source']))
        self.adapter.rerank = bury
        plan = ContextPlan('policy_question', retrieval_query='Webex seat Finance approval',
                           required_evidence=['saas_renewal_freeze_q3.md'], requires_retrieval=True)
        result = self.node(plan)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.refinements, 1)
        self.assertTrue(any(e.citation.source == 'memos/saas_renewal_freeze_q3.md' for e in result.evidence))

    def test_reranker_failure_falls_back_to_knn_order(self):
        def broken(query, candidates):
            candidates[0]['text'] = 'mutated before failing'
            raise ValueError('Reranker must score every candidate')
        self.adapter.rerank = broken
        result = self.node(replace(self.plan(), required_evidence=[]))  # one round
        self.assertEqual(result.status, 'ok')
        self.assertEqual(self.adapter.rerank_failures, 1)
        self.assertEqual([e.citation.chunk_id for e in result.evidence],
                         [r['chunk_id'] for r in self.search.filtered[:4]])
        self.assertNotIn('mutated before failing', ' '.join(e.text for e in result.evidence))

    def test_reranker_cannot_mutate_authorized_candidates(self):
        def mutate(query, candidates):
            candidates[0]['text'] = 'invented'
            return candidates
        self.adapter.rerank = mutate
        with self.assertRaises(ValueError):
            self.node(self.plan())

    def test_reranker_cannot_fabricate_evidence(self):
        self.adapter.rerank = lambda q, cs: [dict(c, text='invented') for c in cs]
        with self.assertRaises(ValueError):
            self.node(self.plan())


class IngestionTests(unittest.TestCase):
    def test_allowlist_dates_ids_and_personal_metadata(self):
        first = load_chunks(ROOT)
        self.assertEqual(first, load_chunks(ROOT))
        expected = {p.relative_to(ROOT).as_posix() for folder in ('employment', 'policies', 'it_kb', 'memos')
                    for p in (ROOT / folder).glob('*.md')}
        self.assertEqual({r['source'] for r in first}, expected)
        self.assertFalse(any(r['source'].startswith('database/') for r in first))
        self.assertEqual(len({r['chunk_id'] for r in first}), len(first))
        memo = next(r for r in first if r['source'].endswith('webex_license_cleanup_memo.md'))
        self.assertEqual(memo['updated_at'], '2026-06-25')
        personal = [r for r in first if r['collection'] == 'employment']
        self.assertTrue(all(r['audience'] == ['UG_HR'] for r in personal))

    def test_new_unreviewed_document_is_not_silently_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'employment').mkdir()
            (root / 'employment' / 'unknown.md').write_text('Private employee information')
            with self.assertRaisesRegex(ValueError, 'Missing reviewed'):
                load_chunks(root)

    def test_contract_support_and_root_independent_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'contracts').mkdir()
            (root / 'contracts/webex.md').write_text('Webex subscription limit: 40 seats.')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'contracts/webex.md': {
                'audience': ['UG_HR', 'UG_IT'], 'sensitivity': 'confidential',
                'subject_employee_id': None, 'self_service': False, 'allowed_employee_ids': [],
            }}))
            records = load_chunks(root, manifest_path=manifest)
            self.assertEqual(records[0]['collection'], 'contracts')
            other = root / 'other'
            (other / 'contracts').mkdir(parents=True)
            (other / 'contracts/webex.md').write_text('Webex subscription limit: 40 seats.')
            self.assertEqual(records, load_chunks(other, manifest_path=manifest))

    def test_missing_subject_and_regular_audience_rejected(self):
        from maya.ingestion import validate_metadata
        meta = {'audience': ['UG_REGULAR'], 'sensitivity': 'restricted',
                'subject_employee_id': 'E001', 'self_service': True, 'allowed_employee_ids': ['E001']}
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_index_mapping_supports_filter_and_dates(self):
        body = index_body(1024)
        props = body['mappings']['properties']
        self.assertEqual(props['vector']['method']['engine'], 'faiss')
        self.assertEqual(props['audience']['type'], 'keyword')
        self.assertEqual(props['updated_at']['type'], 'date')
        self.assertEqual(index_body(1024, serverless=True)['mappings']['properties']['vector']['method']['engine'], 'faiss')
        self.assertNotIn('engine', index_body(1024, serverless=True, engine='auto')['mappings']['properties']['vector']['method'])
        with self.assertRaises(ValueError):
            index_body(1024, engine='nmslib')

    def test_unchanged_vectors_skip_embedding_and_bulk(self):
        records = load_chunks(ROOT)
        client = Mock()
        client.count.return_value = {'count': len(records)}
        client.search.return_value = {'hits': {'hits': [
            {'_source': dict(r, vector=[1, 0])} for r in records
        ]}}
        embed = Mock()
        self.assertEqual(ingest(client, index='x', records=records, embed=embed, dimension=2), 0)
        embed.assert_not_called()
        client.bulk.assert_not_called()
        client.indices.delete.assert_not_called()

    def test_changed_text_permissions_or_missing_vectors_never_skip(self):
        records = load_chunks(ROOT)[:1]
        for change in ('text', 'audience', 'vector'):
            with self.subTest(change=change):
                stored = dict(records[0], vector=[1, 0])
                stored[change] = {'text': 'changed', 'audience': ['UG_REGULAR'], 'vector': []}[change]
                client = Mock()
                client.count.return_value = {'count': 1}
                client.search.return_value = {'hits': {'hits': [{'_source': stored}]}}
                embed = Mock()
                with self.assertRaisesRegex(ValueError, '--force'):
                    ingest(client, index='x', records=records, embed=embed, dimension=2)
                embed.assert_not_called()
                client.bulk.assert_not_called()

    def test_force_reembeds_and_replaces_only_index_after_embedding(self):
        records = load_chunks(ROOT)[:2]
        events = []
        client = Mock()
        client.count.return_value = {'count': 2}
        client.indices.delete.side_effect = lambda **kw: events.append('delete')
        client.indices.create.side_effect = lambda **kw: events.append('create')
        client.bulk.return_value = {'errors': False}
        def embed(text):
            events.append('embed')
            return [1, 0]
        self.assertEqual(ingest(client, index='maya-test', records=records, embed=embed,
                               dimension=2, force=True, serverless=True), 2)
        self.assertEqual(events, ['embed', 'embed', 'delete', 'create'])
        client.indices.delete.assert_called_once_with(index='maya-test')
        client.search.assert_not_called()
        ops = client.bulk.call_args.kwargs['body']
        self.assertEqual(len(ops), 4)
        self.assertNotIn('_id', ops[0]['index'])

    def test_embedding_failure_preserves_existing_index(self):
        client = Mock()
        client.count.return_value = {'count': 1}
        with self.assertRaises(RuntimeError):
            ingest(client, index='x', records=load_chunks(ROOT)[:1],
                   embed=Mock(side_effect=RuntimeError('embedding failed')), dimension=2, force=True)
        client.indices.delete.assert_not_called()
        client.bulk.assert_not_called()

    def test_embedding_model_change_does_not_skip(self):
        client = Mock()
        client.count.return_value = {'count': 1}
        client.indices.get_mapping.return_value = {'x': {'mappings': {
            '_meta': {'embedding_model_id': 'old-model'}}}}
        embed = Mock()
        with self.assertRaisesRegex(ValueError, '--force'):
            ingest(client, index='x', records=load_chunks(ROOT)[:1], embed=embed,
                   dimension=2, embedding_model_id='new-model')
        embed.assert_not_called()

    def test_ingest_empty_index_only_and_bulk_errors_surface(self):
        client = Mock()
        client.count.return_value = {'count': 1}
        with self.assertRaises(ValueError):
            ingest(client, index='x', records=[], embed=Mock(), dimension=2)
        client.bulk.assert_not_called()
        client.count.return_value = {'count': 0}
        client.bulk.return_value = {'errors': False}
        records = load_chunks(ROOT)
        self.assertEqual(ingest(client, index='x', records=records, embed=lambda t: [1, 0], dimension=2), len(records))
        ops = client.bulk.call_args.kwargs['body']
        self.assertEqual(ops[1]['chunk_id'], records[0]['chunk_id'])
        self.assertNotIn('_id', ops[0]['index'])
        client.bulk.return_value = {'errors': True}
        with self.assertRaises(RuntimeError):
            ingest(client, index='x', records=records[:1], embed=lambda t: [1, 0], dimension=2)


class RerankerTests(unittest.TestCase):
    def test_forced_tool_scores_order_candidates(self):
        client = Mock()
        client.converse.return_value = {'output': {'message': {'content': [{'toolUse': {
            'name': 'rank', 'input': {'scores': [{'index': 0, 'score': .1}, {'index': 1, 'score': .9}]},
        }}]}}}
        candidates = [{'text': 'first', 'chunk_id': 'a'}, {'text': 'second', 'chunk_id': 'b'}]
        reranker = BedrockReranker(client, model_id='test-model')
        self.assertEqual(reranker('question', candidates), list(reversed(candidates)))
        self.assertEqual(client.converse.call_args.kwargs['toolConfig']['toolChoice'], {'tool': {'name': 'rank'}})
        client.converse.return_value['output']['message']['content'][0]['toolUse']['input']['scores'].pop()
        with self.assertRaises(ValueError):
            reranker('question', candidates)


if __name__ == '__main__':
    unittest.main()


class BackendEndpointTests(unittest.TestCase):
    def test_endpoint_override_and_collection_lookup(self):
        from unittest.mock import patch
        from maya.backend import resolve_endpoint
        session = Mock()
        session.client.return_value.batch_get_collection.return_value = {
            'collectionDetails': [{'status': 'ACTIVE', 'collectionEndpoint': 'https://example.aoss.amazonaws.com'}]}
        with patch.dict('os.environ', {'OPENSEARCH_ENDPOINT': 'https://override.example'}, clear=True):
            self.assertEqual(resolve_endpoint(session, service='aoss'), 'https://override.example')
            session.client.assert_not_called()
        with patch.dict('os.environ', {'OPENSEARCH_COLLECTION': 'maya-test'}, clear=True):
            self.assertEqual(resolve_endpoint(session, service='aoss'), 'https://example.aoss.amazonaws.com')
            session.client.return_value.batch_get_collection.assert_called_once_with(names=['maya-test'])
