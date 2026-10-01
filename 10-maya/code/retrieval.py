"""Hard-filtered OpenSearch retrieval and a deterministic runtime-config node."""
import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Protocol

from .access import (COLLECTIONS, DENIAL, PermissionDenied, authorize_subject, caller_from_config,
                     hard_access_filter, record_is_allowed, subject_filter)
from .schemas import CallerContext, ContextPlan, Evidence, EvidenceCitation

SOURCE_FIELDS = ['text', 'source', 'source_id', 'chunk_id', 'collection', 'audience',
                 'subject_employee_id', 'sensitivity', 'updated_at', 'self_service',
                 'allowed_employee_ids']


class EvidenceRetriever(Protocol):
    async def retrieve(self, *, caller: CallerContext, plan: ContextPlan,
                       sources: tuple[str, ...] = ()) -> list[Evidence]:
        """One authorized retrieval round. Refinement belongs to application code.

        `sources` narrows a refinement to named required documents; it is ANDed with,
        never a replacement for, the caller's hard access filter.
        """
        ...


def source_filter(sources) -> dict:
    """Exact keyword match on required documents: filename, path or stable source ID."""
    paths = sorted({name if '/' in name else f'{collection}/{name}'
                    for name in sources for collection in COLLECTIONS})
    return {'bool': {'should': [{'terms': {'source': paths}},
                                {'terms': {'source_id': sorted(sources)}}],
                     'minimum_should_match': 1}}


def query_for(plan: ContextPlan) -> str:
    query = plan.retrieval_query or ' '.join([
        plan.current_intent, *plan.relevant_facts, *plan.active_constraints,
        *plan.required_evidence,
    ])
    # Preserve exact requirements even when the planner's natural-language query
    # omitted them; they are selection criteria, not optional prompt decoration.
    missing = [source for source in plan.required_evidence if source not in query]
    return query + ('\nRequired sources: ' + ' '.join(missing) if missing else '')


class OpenSearchEvidenceRetriever:
    def __init__(self, client, *, index: str, embed, rerank,
                 candidate_count: int = 20, evidence_count: int = 4):
        if not 0 < evidence_count < candidate_count:
            raise ValueError('Retrieve wider than the final evidence set')
        self.client, self.index = client, index
        self.embed, self.rerank = embed, rerank
        self.candidate_count, self.evidence_count = candidate_count, evidence_count
        self.rerank_failures = 0

    async def retrieve(self, *, caller: CallerContext, plan: ContextPlan,
                       sources: tuple[str, ...] = ()) -> list[Evidence]:
        authorize_subject(caller, plan.subject_employee_id)
        if not plan.requires_retrieval:
            return []
        return await asyncio.to_thread(self._retrieve, caller, plan, tuple(sources))

    def _retrieve(self, caller: CallerContext, plan: ContextPlan, sources=()) -> list[Evidence]:
        query = query_for(plan)
        # Always inside knn, never post_filter. Soft scope is ANDed with immutable ACL.
        clauses = [hard_access_filter(caller), subject_filter(plan.subject_employee_id)]
        if sources:
            # A named-document refinement: embeddings barely respond to filenames.
            clauses.append(source_filter(sources))
        filt = {'bool': {'must': clauses}}
        body = {'size': self.candidate_count,
                'query': {'knn': {'vector': {'vector': self.embed(query),
                          'k': self.candidate_count, 'filter': filt}}},
                '_source': SOURCE_FIELDS}
        response = self.client.search(index=self.index, body=body)
        hits = response.get('hits', {}).get('hits') if isinstance(response, dict) else None
        if not isinstance(hits, list):
            raise ValueError('Malformed OpenSearch response')
        candidates = []
        seen = set()
        for hit in hits:
            # A hit without _source goes through the same fail-closed ACL check.
            record = hit.get('_source') if isinstance(hit, dict) else None
            if not record_is_allowed(record, caller, plan.subject_employee_id):
                raise PermissionDenied()
            if record['chunk_id'] not in seen:
                candidates.append(record)
                seen.add(record['chunk_id'])
        if not candidates:
            return []
        originals = {c['chunk_id']: deepcopy(c) for c in candidates}
        try:
            ranked = self.rerank(query, candidates)
        except Exception:
            # Reranking only reorders ACL-checked candidates. An invalid or failed
            # ranking falls back to k-NN order instead of discarding the round.
            self.rerank_failures += 1
            ranked = [deepcopy(c) for c in originals.values()]
        # The reranker may reorder candidates, but cannot invent content or citations.
        if (len(ranked) != len(candidates)
                or len({c['chunk_id'] for c in ranked}) != len(candidates)
                or any(c != originals.get(c['chunk_id']) for c in ranked)):
            raise ValueError('Reranker must return a permutation of authorized candidates')
        limit = self.evidence_count
        if sources:
            # A named-document refinement covers each document with its best passage
            # first; the set stays bounded by the number of named documents.
            best, seen_sources = [], set()
            for c in ranked:
                if c['source'] not in seen_sources:
                    best.append(c)
                    seen_sources.add(c['source'])
            ranked = best + [c for c in ranked if c not in best]
            limit = min(self.candidate_count, max(self.evidence_count, len(best)))
        return [Evidence(c['text'], EvidenceCitation(
            source=c['source'], chunk_id=c['chunk_id'], source_id=c['source_id'],
            collection=c['collection'], audience=tuple(c['audience']),
            subject_employee_id=c.get('subject_employee_id'),
            sensitivity=c['sensitivity'], updated_at=c.get('updated_at'),
        )) for c in ranked[:limit]]


def missing_evidence(plan: ContextPlan, evidence: list[Evidence]) -> list[str]:
    """Requirements are source paths, filenames or stable source IDs, not prose.

    Source presence is a deterministic coverage check, not proof that a passage
    answers a factual question; later checklist validation must still check claims.
    """
    present = {name for e in evidence for name in (
        e.citation.source, PurePosixPath(e.citation.source).name, e.citation.source_id)}
    return [name for name in plan.required_evidence if name not in present]


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    evidence: tuple[Evidence, ...] = ()
    missing: tuple[str, ...] = ()
    refinements: int = 0
    message: str = ''


async def retrieve_evidence_node(plan: ContextPlan, config: dict,
                                 retriever: EvidenceRetriever) -> RetrievalResult:
    """Milestone 3 can wire this node directly; caller never comes from graph state.

    Per invocation (one turn): initial round plus at most one deterministic refinement.
    Denial returns no plan, document names, employee facts, counts or citations.
    """
    try:
        caller = caller_from_config(config)
        authorize_subject(caller, plan.subject_employee_id)
        if not plan.requires_retrieval:
            return RetrievalResult('skipped')
        evidence = await retriever.retrieve(caller=caller, plan=plan)
        missing = missing_evidence(plan, evidence)
        refinements = 0
        if missing:
            refined = replace(plan, retrieval_query=f'{query_for(plan)}\nRequired sources: ' + ' '.join(missing))
            extra = await retriever.retrieve(caller=caller, plan=refined, sources=tuple(missing))
            refinements = 1
            # Preserve first-round useful evidence; two rounds remain a bounded set.
            evidence = list({e.citation.chunk_id: e for e in [*evidence, *extra]}.values())
            missing = missing_evidence(plan, evidence)
        return RetrievalResult('unresolved' if missing else 'ok', tuple(evidence),
                               tuple(missing), refinements)
    except PermissionDenied:
        return RetrievalResult('denied', message=DENIAL)
