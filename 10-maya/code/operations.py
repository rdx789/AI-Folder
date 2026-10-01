"""Read schemas and caller guards shared by local and MCP operational adapters."""
import asyncio
import json
import re
from typing import Protocol

from .access import PermissionDenied, authorize_subject
from .policy import READ_TOOLS
from .schemas import CallerContext, ContextPlan, ReadToolCall, Evidence, EvidenceCitation

PARAMETERS = {
    'get_employee': {'query': 'Employee ID or full name'},
    'list_onboarding_tasks': {'employee_id': 'Employee ID'},
    'list_employee_tickets': {'employee_id': 'Employee ID', 'status': 'Optional status'},
    'check_asset_inventory': {'asset_type': 'Optional asset type', 'location': 'Optional location'},
    'check_software_subscription': {'software': 'Software name'},
    'list_policies': {},
    'get_policy': {'name': 'Exact policy name'},
}
REQUIRED = {
    'get_employee': ['query'], 'list_onboarding_tasks': ['employee_id'],
    'list_employee_tickets': ['employee_id'], 'check_software_subscription': ['software'],
    'get_policy': ['name'],
}


def tool_schemas(names):
    from .paths import DATA_PATH
    policy_names = sorted(p.stem for p in (DATA_PATH / 'policies').glob('*.md'))
    result = tuple({'name': name, 'description': name.replace('_', ' '),
                  'input_schema': {'type': 'object', 'properties': {
                      k: {'type': 'string', 'description': v} for k, v in PARAMETERS[name].items()},
                      'required': REQUIRED.get(name, []), 'additionalProperties': False}}
                 for name in names)
    for schema in result:
        if schema['name'] == 'get_policy':
            schema['input_schema']['properties']['name'].update(
                enum=policy_names, description='Exact policy stem; no .md suffix. Memos use EvidenceRetriever.')
    return result


class ReadPort(Protocol):
    async def call(self, name: str, arguments: dict[str, str]): ...


class LocalReadPort:
    """Local development adapter for the same seven read-only MCP functions."""
    async def call(self, name, arguments):
        if name not in READ_TOOLS:
            raise PermissionDenied()
        from .server import db
        if name in ('get_policy', 'list_policies'):
            from .server.server import get_policy, list_policies
            fn = {'get_policy': get_policy, 'list_policies': list_policies}[name]
        else:
            fn = getattr(db, name)
        return await asyncio.to_thread(fn, **arguments)


class MCPReadPort:
    """Injected, already-connected MCP ClientSession; no transport/auth side effects."""
    def __init__(self, session):
        self.session = session

    async def call(self, name, arguments):
        if name not in READ_TOOLS:
            raise PermissionDenied()
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            raise ValueError('Operational read failed')
        if result.structuredContent is not None:
            value = result.structuredContent
            return value.get('result', value) if isinstance(value, dict) else value
        content = '\n'.join(c.text for c in result.content if c.type == 'text')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content


def authorize_read(caller: CallerContext, plan: ContextPlan, call: ReadToolCall,
                   selected: tuple[str, ...], newest_message: str):
    authorize_subject(caller, 'E001')
    if call.name not in selected or call.name not in READ_TOOLS:
        raise PermissionDenied()
    args = call.arguments
    if (not isinstance(args, dict) or set(args) - set(PARAMETERS[call.name])
            or not set(REQUIRED.get(call.name, [])) <= set(args)
            or any(not isinstance(v, str) for v in args.values())):
        raise ValueError('Invalid operational read arguments')
    # A tangent is permitted only in its active turn. Never use a model-selected
    # subject to widen retrieval or to authorize a later employee read.
    target = 'E010' if (plan.current_intent == 'ticket_status'
                        and 'rachel' in newest_message.lower()) else 'E001'
    authorize_subject(caller, target)
    if call.name in ('list_employee_tickets', 'list_onboarding_tasks'):
        if args['employee_id'] != target or (call.name == 'list_onboarding_tasks' and target != 'E001'):
            raise PermissionDenied()
    if call.name == 'get_employee':
        allowed = {target.lower(), 'rachel stein' if target == 'E010' else 'maya cohen'}
        if args['query'].lower() not in allowed:
            raise PermissionDenied()
    if call.name == 'list_employee_tickets' and plan.current_intent != 'ticket_status':
        raise PermissionDenied()


def sanitize_result(name, value, caller, target='E001'):
    if name == 'get_employee':
        # Malformed rows fail closed like foreign rows; never AttributeError mid-read.
        if not isinstance(value, list) or any(
                not isinstance(r, dict) or r.get('employee_id') != target for r in value):
            raise PermissionDenied()
    if name == 'check_asset_inventory' and isinstance(value, list):
        # Inventory does not expose other employees' assignments to self-service.
        if caller.user_group != 'UG_HR':
            value = [r for r in value if isinstance(r, dict)
                     and r.get('employee_id') in (None, '', caller.employee_id)]
    return value


def operational_evidence(name, arguments, value, *, turn: int, target='E001'):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    # These are record snapshots, explicitly distinct from retrieved documents.
    is_policy = name == 'get_policy'
    dated = re.search(r'^(?:Effective date|Last updated|Date):\s*(\d{4}-\d{2}-\d{2})', text, re.M)
    citation = EvidenceCitation(
        source=f"policies/{arguments['name']}.md" if is_policy else 'novaops/' + name,
        chunk_id=f'read:{turn}:{name}:{json.dumps(arguments, sort_keys=True)}',
        section='authorized operational snapshot', collection='operational',
        subject_employee_id=target if name in ('get_employee', 'list_onboarding_tasks', 'list_employee_tickets') else None,
    )
    if is_policy:
        from dataclasses import replace
        citation = replace(citation, collection='policies', section='full policy',
                           updated_at=dated[1] if dated else None)
    return Evidence(text, citation)
