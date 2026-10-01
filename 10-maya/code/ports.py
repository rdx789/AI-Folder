"""Milestone 4 boundary. Maya never exposes this operation as a model tool."""

import asyncio
import hashlib
import json
from dataclasses import asdict
from typing import Protocol

from .access import authorize_subject
from .schemas import CallerContext, PendingAccessResult, WebexHandoff, WebexDispatchRecord


class WebexPort(Protocol):
    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult: ...


def webex_request_key(*, thread_id: str, caller: CallerContext, business_reason: str) -> str:
    """Stable request identity, independent of model text, turn count and citations."""
    payload = {'version': 1, 'thread_id': thread_id, 'caller': asdict(caller),
               'employee_id': 'E001', 'system': 'Webex', 'business_reason': business_reason}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_handoff(handoff: WebexHandoff) -> None:
    if not isinstance(handoff, WebexHandoff):
        raise ValueError('Expected a typed Webex handoff')
    authorize_subject(handoff.caller, 'E001')
    if (handoff.employee_id != 'E001' or handoff.system != 'Webex' or handoff.kind != 'webex_access'
            or not isinstance(handoff.business_reason, str) or not handoff.business_reason.strip()
            or not isinstance(handoff.idempotency_key, str) or not handoff.idempotency_key):
        raise ValueError('Invalid Webex request')


def validate_pending_result(result: PendingAccessResult) -> PendingAccessResult:
    # A model/dict claiming "granted" must not be coerced into a pending dataclass.
    if (type(result) is not PendingAccessResult or result.status != 'pending'
            or not isinstance(result.request_id, str) or not result.request_id.strip()):
        raise ValueError('Webex port must return a typed pending acknowledgement')
    return result


class FakeWebexPort:
    """Homework fake: every invocation is recorded; no approval or access write."""

    def __init__(self):
        self.calls: list[WebexHandoff] = []

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        validate_handoff(handoff)
        self.calls.append(handoff)
        return PendingAccessResult('WEBEX-' + handoff.idempotency_key[:20])


class WebexDispatchGuard:
    """Runtime protection for node retry/checkpoint-save failure in one process.

    Thread checkpoints persist completed/uncertain attempts across graph rebuilds.
    This guard additionally remembers an attempt immediately, before awaiting the
    injected port. An ambiguous exception never automatically retries a write.
    """

    def __init__(self, port: WebexPort | None):
        self.port = port
        self._records: dict[str, WebexDispatchRecord] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def request_access(self, handoff: WebexHandoff) -> PendingAccessResult:
        validate_handoff(handoff)
        if self.port is None:
            raise ValueError('Webex workflow is not configured')
        key = handoff.idempotency_key
        async with self._locks.setdefault(key, asyncio.Lock()):
            record = self._records.get(key)
            if record:
                # Evidence may refresh; business request identity must not change.
                if (record.handoff.caller, record.handoff.employee_id, record.handoff.system,
                    record.handoff.business_reason) != (handoff.caller, handoff.employee_id,
                                                       handoff.system, handoff.business_reason):
                    raise ValueError('Idempotency key reused for a different request')
                if record.status == 'pending':
                    return record.result
                raise ValueError('Dispatch acknowledgement unresolved; reconciliation required')
            record = WebexDispatchRecord(handoff)
            self._records[key] = record
            try:
                result = validate_pending_result(await self.port.request_access(handoff))
            except BaseException:
                record.status = 'unresolved'
                raise
            record.status, record.result = 'pending', result
            return result
