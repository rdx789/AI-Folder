"""Offline test doubles over the actual corpus and operational records.

The model classifies request text and extracts supplied evidence. It never reads
sessions.json, turn numbers, expected tools or expected answers. This exercises
workflow contracts, not a provider's semantic quality or live OpenSearch ranking.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re

from ..answer_context import (AnswerSelection, candidate_catalogue, focused_evidence, resolve_selection,
                              scoped_catalogue)
from ..graph import Dependencies, handoff_requested, recall_only
from ..policy import full_checklist_request, necessary_reads, resolve_pending_reads
from ..paths import DATA_PATH
from ..ingestion import load_chunks
from ..operations import LocalReadPort
from ..ports import FakeWebexPort
from ..retrieval import OpenSearchEvidenceRetriever
from ..schemas import ChecklistTask, ContextPlan, ModelTurn, ReadToolCall
from .capture import build_recorded_graph



def matches(query, record):
    """Independent evaluator: apply ACL/subject filter before candidate ranking."""
    if 'term' in query:
        key, value = next(iter(query['term'].items()))
        actual = record.get(key)
        return value in actual if isinstance(actual, list) else value == actual
    if 'terms' in query:
        key, values = next(iter(query['terms'].items()))
        actual = record.get(key)
        return bool(set(actual) & set(values)) if isinstance(actual, list) else actual in values
    if 'exists' in query:
        return record.get(query['exists']['field']) is not None
    spec = query['bool']
    return (all(matches(q, record) for q in spec.get('must', []))
            and not any(matches(q, record) for q in spec.get('must_not', []))
            and sum(matches(q, record) for q in spec.get('should', [])) >= spec.get('minimum_should_match', 0))


def score(query, record):
    filename = record['source'].rsplit('/', 1)[-1]
    words = set(re.findall(r'[a-z0-9]+', query.lower()))
    text_words = set(re.findall(r'[a-z0-9]+', record['text'].lower()))
    return (1000 if filename in query else 0) + len(words & text_words)


class OfflineSearch:
    def __init__(self, records):
        self.records, self.queries = records, {}
        self.embeddings, self.searches, self.reranks = 0, 0, 0

    def embed(self, query):
        self.embeddings += 1
        digest = hashlib.sha256(query.encode()).digest()
        vector = [int.from_bytes(digest[:4], 'big'), int.from_bytes(digest[4:8], 'big')]
        self.queries[tuple(vector)] = query
        return vector

    def search(self, *, index, body):
        self.searches += 1
        knn = body['query']['knn']['vector']
        query = self.queries[tuple(knn['vector'])]
        eligible = [r for r in self.records if matches(knn['filter'], r)]
        ranked = sorted(eligible, key=lambda r: (-score(query, r), r['source'], r['chunk_id']))
        return {'hits': {'hits': [{'_source': r} for r in ranked[:knn['k']]]}}

    def rerank(self, query, candidates):
        self.reranks += 1
        return sorted(candidates, key=lambda r: (-score(query, r), r['source'], r['chunk_id']))

    def counts(self):
        return {'embeddings': self.embeddings, 'searches': self.searches, 'reranks': self.reranks}


class OfflineMayaModel:
    """Rule-based provider substitute; decisions depend on inputs, not eval labels."""

    async def distill(self, *, memory, fresh_history):
        return deepcopy(memory)  # graph already pins user facts and pending decisions

    async def plan(self, *, newest_message, memory, fresh_history):
        text = newest_message.lower()
        if recall_only(newest_message):
            return ContextPlan('recall')
        if handoff_requested(newest_message):
            return ContextPlan('access_request')
        if 'employee record' in text:
            return ContextPlan('employee_lookup', requires_operational_reads=True)
        if 'offer letter' in text:
            sources = ['maya_cohen_offer_letter.md', 'maya_cohen_onboarding_summary.md']
            return ContextPlan('onboarding_status', required_evidence=sources,
                               retrieval_query=' '.join(sources), requires_retrieval=True)
        if 'ticket' in text and 'rachel' in text:
            return ContextPlan('ticket_status', requires_operational_reads=True)
        if 'checklist' in text:
            return ContextPlan('onboarding_status', requires_operational_reads=True)
        if 'laptop' in text:
            return ContextPlan('equipment_request', requires_operational_reads=True)
        if 'monitor' in text or 'headset' in text:
            return ContextPlan('equipment_request', requires_operational_reads=True)
        if 'finance' in text or 'freeze' in text:
            sources = ['saas_renewal_freeze_q3.md', 'webex_license_cleanup_memo.md']
            return ContextPlan('policy_question', required_evidence=sources,
                               retrieval_query=' '.join(sources), requires_retrieval=True)
        if 'webex' in text and 'seat' in text:
            sources = ['webex_license_assignment.md', 'webex_license_cleanup_memo.md']
            return ContextPlan('subscription_review', requires_operational_reads=True,
                               required_evidence=sources, retrieval_query=' '.join(sources), requires_retrieval=True)
        return ContextPlan('other')

    async def respond(self, *, newest_message, plan, memory, history, evidence, tools, previous_checklist):
        if full_checklist_request(newest_message):
            return full_checklist_turn(newest_message, plan, history, evidence, tools)
        text = newest_message.lower()
        # Decide necessary reads from intent and question, then check the current
        # exchange. Cached records never satisfy an explicit refresh request.
        calls = []
        if plan.requires_operational_reads:
            if plan.current_intent == 'employee_lookup':
                calls = [ReadToolCall('get_employee', {'query': 'Maya Cohen'})]
            elif plan.current_intent == 'ticket_status':
                calls = [ReadToolCall('list_employee_tickets', {'employee_id': 'E010'})]
            elif 'checklist' in text:
                calls = [ReadToolCall('list_onboarding_tasks', {'employee_id': 'E001'})]
            elif 'laptop' in text:
                calls = [ReadToolCall('check_asset_inventory', {'asset_type': 'Laptop', 'location': memory.important_facts.get('location', '')})]
            elif 'monitor' in text or 'headset' in text:
                calls = [ReadToolCall('get_policy', {'name': 'equipment_policy'})]
            elif plan.current_intent == 'subscription_review':
                calls = [ReadToolCall('check_software_subscription', {'software': 'Webex'})]
        current = list(history)
        start = max((i for i, m in enumerate(current) if m.role == 'user'), default=0)
        done = {name for m in current[start:] if m.role == 'assistant' for name in m.tool_calls}
        pending = [c for c in calls if c.name not in done]
        if pending:
            available = {t['name'] for t in tools}
            return ModelTurn(tool_calls=[c for c in pending if c.name in available])
        return ModelTurn(tasks=extract_tasks(evidence, plan.current_intent))


def full_checklist_turn(newest, plan, history, evidence, tools):
    """Stand-in for a model that selects every supported statement on a full checklist."""
    start = max((i for i, m in enumerate(history) if m.role == 'user'), default=0)
    done = {name for m in history[start:] for name in m.tool_calls}
    available = {t['name'] for t in tools}
    pending = [n for n in necessary_reads(plan, newest) if n not in done and n in available]
    calls = resolve_pending_reads(pending, tools, newest, plan, evidence) if pending else []
    if calls:
        return ModelTurn(tool_calls=calls)
    focus = focused_evidence(evidence, plan, newest)
    candidates = scoped_catalogue(candidate_catalogue(focus), newest, focus)
    return ModelTurn(tasks=resolve_selection(AnswerSelection(candidate_ids=list(candidates)), candidates, focus))


def extract_tasks(evidence, intent):
    tasks = []
    for item in evidence:
        citation, text = item.citation, item.text
        if citation.collection == 'operational':
            try:
                rows = json.loads(text)
            except ValueError:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                quote = json.dumps(row, sort_keys=True)
                source = citation.source.rsplit('/', 1)[-1]
                status = {'completed':'complete','pending':'pending','planned':'proposed','blocked':'blocked','open':'pending','available':'proposed'}.get(row.get('status'))
                if source == 'list_onboarding_tasks':
                    description, key = row['description'], 'onboarding:' + row['task_id']
                elif source == 'check_asset_inventory' and row.get('status') == 'available':
                    description, key = row['model'], 'asset:' + row['asset_id']
                elif source == 'list_employee_tickets' and intent == 'ticket_status':
                    description, key = quote, 'ticket:' + row['ticket_id']
                elif source == 'check_software_subscription' and row['active_seats'] >= row['seat_limit']:
                    description, key, status = quote, 'capacity:' + row['system_name'], 'blocked'
                else:
                    continue
                if status:
                    tasks.append(ChecklistTask(description, status, (citation,), key, (quote,)))
            continue
        # Document excerpts preserve actual words. Do not turn a conditional Webex
        # warning into an asserted blocker, or old runbook advice into a new task.
        normalized = ' '.join(text.split())
        patterns = [
            (r'Required systems:.*?(?= Equipment entitlement:| Special notes:|$)', 'proposed', 'required:systems'),
            (r'Equipment entitlement:.*?(?= Special notes:|$)', 'proposed', 'required:equipment'),
            (r'For Q3 2026,.*?require Finance approval\.', 'pending', 'finance:threshold'),
            (r'Webex is at.*?IT should not assign.*?Finance approves expansion\.', 'blocked', 'rule:webex-capacity'),
        ]
        for pattern, status, key in patterns:
            match = re.search(pattern, normalized)
            if match:
                excerpt = match.group()
                tasks.append(ChecklistTask(excerpt, status, (citation,), key, (excerpt,)))
        if citation.source.endswith('equipment_policy.md'):
            normalized = ' '.join(text.split())
            for pattern, key in [(r'- Remote or hybrid.*?(?= - | ## )','guidance:monitor'),
                                 (r'- Customer Success Managers.*?(?= ## )','guidance:headset')]:
                match = re.search(pattern, normalized)
                if match:
                    excerpt = match.group()
                    tasks.append(ChecklistTask(excerpt, 'proposed', (citation,), key, (excerpt,)))
    return tasks


@dataclass
class OfflineRuntime:
    app: object
    capture: object
    webex: FakeWebexPort
    search: OfflineSearch


def build_offline_runtime():
    search = OfflineSearch(load_chunks(DATA_PATH))
    retriever = OpenSearchEvidenceRetriever(search, index='offline', embed=search.embed, rerank=search.rerank)
    fake = FakeWebexPort()
    app, capture = build_recorded_graph(Dependencies(OfflineMayaModel(), retriever, fake, LocalReadPort()))
    return OfflineRuntime(app, capture, fake, search)
