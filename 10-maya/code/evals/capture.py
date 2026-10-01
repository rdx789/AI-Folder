"""Dependency-boundary records used by both offline and injected live runs."""
from copy import deepcopy
from dataclasses import dataclass, field, asdict

from ..graph import Dependencies, build_graph


@dataclass
class Capture:
    plans: list[dict] = field(default_factory=list)
    models: list[dict] = field(default_factory=list)
    retrieval: list[dict] = field(default_factory=list)
    reads: list[dict] = field(default_factory=list)
    handoffs: list[dict] = field(default_factory=list)
    distillations: list[dict] = field(default_factory=list)
    llm: list[dict] = field(default_factory=list)  # filled by evals.metering

    def position(self):
        return {k: len(getattr(self, k)) for k in self.__dataclass_fields__}

    def since(self, position):
        return {k: deepcopy(getattr(self, k)[position[k]:]) for k in self.__dataclass_fields__}


class RecordedModel:
    def __init__(self, model, capture): self.model, self.capture = model, capture

    async def plan(self, **kwargs):
        record = {'newest_message': kwargs['newest_message']}
        self.capture.plans.append(record)
        result = await self.model.plan(**kwargs)
        record['plan'] = asdict(result)
        return result

    async def distill(self, **kwargs):
        record = {'folded_messages': len(kwargs['fresh_history'])}
        self.capture.distillations.append(record)
        result = await self.model.distill(**kwargs)
        record['memory'] = asdict(result)
        return result

    async def respond(self, **kwargs):
        record = {'schemas': deepcopy(kwargs['tools']), 'requested_calls': []}
        self.capture.models.append(record)
        result = await self.model.respond(**kwargs)
        record['draft'] = asdict(result)
        record['requested_calls'] = [asdict(c) for c in result.tool_calls]
        return result


class RecordedRetriever:
    def __init__(self, retriever, capture): self.retriever, self.capture = retriever, capture

    async def retrieve(self, **kwargs):
        record = {'caller': asdict(kwargs['caller']), 'plan': asdict(kwargs['plan']),
                  'sources': list(kwargs.get('sources', ())), 'evidence': []}
        self.capture.retrieval.append(record)
        try:
            result = await self.retriever.retrieve(**kwargs)
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)[:600]
            raise
        record['evidence'] = [asdict(e) for e in result]
        return result


class RecordedReads:
    def __init__(self, port, capture): self.port, self.capture = port, capture

    async def call(self, name, arguments):
        record = {'name': name, 'arguments': deepcopy(arguments)}
        self.capture.reads.append(record)
        try:
            result = await self.port.call(name, arguments)
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)[:600]
            raise
        record['result'] = deepcopy(result)
        return result


class RecordedWebex:
    def __init__(self, port, capture): self.port, self.capture = port, capture

    async def request_access(self, handoff):
        record = {'handoff': asdict(handoff)}
        self.capture.handoffs.append(record)
        try:
            result = await self.port.request_access(handoff)
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)[:600]
            raise
        record['result'] = asdict(result)
        return result


def build_recorded_graph(dependencies, *, checkpointer=None):
    capture = Capture()
    wrapped = Dependencies(
        RecordedModel(dependencies.model, capture), RecordedRetriever(dependencies.retriever, capture),
        RecordedWebex(dependencies.webex, capture) if dependencies.webex is not None else None,
        RecordedReads(dependencies.reads, capture))
    return build_graph(wrapped, checkpointer=checkpointer), capture
