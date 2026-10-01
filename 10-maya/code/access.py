"""Application-owned NovaOps permissions; no model-controlled role arguments."""
import re
from collections.abc import Mapping

from .schemas import CallerContext

GROUPS = frozenset({'UG_HR', 'UG_IT', 'UG_REGULAR'})
COLLECTIONS = ('employment', 'policies', 'it_kb', 'contracts', 'memos')
DENIAL = "I can't share that information. Please contact HR for assistance."


class PermissionDenied(Exception):
    def __init__(self):
        super().__init__(DENIAL)


def validate_caller(caller: CallerContext) -> None:
    if (not isinstance(caller, CallerContext)
            or not isinstance(caller.employee_id, str)
            or not re.fullmatch(r'E\d{3}', caller.employee_id)
            or not isinstance(caller.user_group, str)
            or caller.user_group not in GROUPS):
        raise PermissionDenied()


def caller_from_config(config: Mapping) -> CallerContext:
    """Only the trusted host sets configurable.caller; never copy it from state/input.

    This boundary consumes an authenticated principal, it does not authenticate it.
    A public API must construct this configuration itself, not accept it from clients.
    """
    if not isinstance(config, Mapping) or not isinstance(config.get('configurable', {}), Mapping):
        raise PermissionDenied()
    caller = config.get('configurable', {}).get('caller')
    validate_caller(caller)
    return caller


def authorize_subject(caller: CallerContext, subject: str) -> None:
    validate_caller(caller)
    if not isinstance(subject, str) or not re.fullmatch(r'E\d{3}', subject):
        raise PermissionDenied()
    if caller.user_group != 'UG_HR' and caller.employee_id != subject:
        raise PermissionDenied()


def hard_access_filter(caller: CallerContext) -> dict:
    validate_caller(caller)
    group_access = {'term': {'audience': caller.user_group}}
    self_access = {'bool': {'must': [
        {'term': {'self_service': True}},
        {'term': {'subject_employee_id': caller.employee_id}},
        {'term': {'allowed_employee_ids': caller.employee_id}},
    ]}}
    return {'bool': {'must': [
        {'terms': {'collection': list(COLLECTIONS)}},
        {'terms': {'sensitivity': ['internal', 'confidential', 'restricted']}},
        {'bool': {'should': [group_access, self_access], 'minimum_should_match': 1}},
    ]}}


def subject_filter(subject: str) -> dict:
    """Soft relevance scope: own/target documents plus non-personal guidance."""
    return {'bool': {'should': [
        {'term': {'subject_employee_id': subject}},
        {'bool': {'must_not': [{'exists': {'field': 'subject_employee_id'}}]}},
    ], 'minimum_should_match': 1}}


def record_is_allowed(record: dict, caller: CallerContext, subject: str) -> bool:
    """Defense before reranker/answer if the backend returns unexpected hits."""
    required = {'source', 'source_id', 'chunk_id', 'text', 'collection', 'audience',
                'sensitivity', 'self_service', 'allowed_employee_ids'}
    if not isinstance(record, dict) or not required <= record.keys():
        return False
    if (not isinstance(record['audience'], list)
            or any(not isinstance(g, str) or g not in GROUPS for g in record['audience'])
            or not isinstance(record['allowed_employee_ids'], list)
            or type(record['self_service']) is not bool
            or not all(isinstance(record[k], str) and record[k] for k in ('text', 'source', 'source_id', 'chunk_id', 'collection', 'sensitivity'))):
        return False
    if record['collection'] not in COLLECTIONS or record['sensitivity'] not in {'internal', 'confidential', 'restricted'}:
        return False
    if record.get('subject_employee_id') not in (None, subject):
        return False
    return caller.user_group in record['audience'] or (
        record['self_service'] is True
        and record.get('subject_employee_id') == caller.employee_id
        and caller.employee_id in record['allowed_employee_ids']
    )
