"""Per-call Bedrock usage, split by the Maya call site; no graph or model changes.

Wraps the injected bedrock-runtime client. Measured values come from Bedrock's own
`usage` / Titan `inputTextTokenCount`. The split of one call's input into prompt,
tool schemas and payload parts is an ESTIMATE by character share of that measured
input; Bedrock does not report those parts separately.
"""
from copy import deepcopy
import io
import json
from threading import Lock
import time

SITES = (('Plan the NEWEST', 'planner'), ('Compress whole', 'distill'),
         ('Execute the current plan', 'answer'))


def call_site(kwargs):
    try:
        tool = kwargs['toolConfig']['tools'][0]['toolSpec']['name']
        system = kwargs['system'][0]['text'].lstrip()
    except (KeyError, IndexError, TypeError):
        return 'other'
    if tool == 'rank':
        return 'rerank'
    return next((site for prefix, site in SITES if system.startswith(prefix)), 'other')


def input_parts(kwargs):
    """Character size of each part of one converse request."""
    parts = {'system_prompt': sum(len(s.get('text', '')) for s in kwargs.get('system', []))}
    parts['output_schema'] = len(json.dumps(kwargs.get('toolConfig', {})))
    text = ''.join(c.get('text', '') for m in kwargs.get('messages', []) for c in m.get('content', []))
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        # Model-visible read-tool schemas travel inside the answer payload.
        parts['read_tool_schemas'] = len(json.dumps(payload.get('read_tools', [])))
        for key in ('candidate_catalogue', 'source_catalogue', 'memory', 'fresh_history'):
            if key in payload:
                parts[key] = len(json.dumps(payload[key], default=str))
        parts['other_payload'] = max(0, len(text) - sum(v for k, v in parts.items()
                                                        if k not in ('system_prompt', 'output_schema')))
    else:
        parts['passages'] = len(text)
    return parts


class MeteredBedrock:
    """Drop-in bedrock-runtime proxy; `sink` is the per-turn capture list."""

    def __init__(self, client, *, keep_requests=True):
        self.client, self.keep_requests = client, keep_requests
        self.sink, self.requests, self.lock = [], [], Lock()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _record(self, record, kwargs):
        with self.lock:
            if self.keep_requests:
                # Exact request, for replaying one decision call under a new prompt.
                self.requests.append({'site': record['site'], 'kwargs': deepcopy(kwargs)})
            self.sink.append(record)

    def converse(self, **kwargs):
        started = time.perf_counter()
        record = {'site': call_site(kwargs), 'model': kwargs.get('modelId')}
        try:
            response = self.client.converse(**kwargs)
        except Exception as exc:
            record.update(error=type(exc).__name__, latency_s=round(time.perf_counter() - started, 3))
            self._record(record, kwargs)
            raise
        usage = response.get('usage', {}) if isinstance(response, dict) else {}
        record.update(input_tokens=usage.get('inputTokens', 0), output_tokens=usage.get('outputTokens', 0),
                      cache_read_tokens=usage.get('cacheReadInputTokens', 0),
                      cache_write_tokens=usage.get('cacheWriteInputTokens', 0),
                      stop_reason=response.get('stopReason'), parts=input_parts(kwargs),
                      latency_s=round(time.perf_counter() - started, 3))
        self._record(record, kwargs)
        return response

    def invoke_model(self, **kwargs):
        started = time.perf_counter()
        response = self.client.invoke_model(**kwargs)
        body = response['body'].read()
        try:
            tokens = json.loads(body).get('inputTextTokenCount', 0)
        except (ValueError, AttributeError):
            tokens = 0
        response['body'] = io.BytesIO(body)  # the caller reads the body again
        self._record({'site': 'embedding', 'model': kwargs.get('modelId'), 'input_tokens': tokens,
                      'output_tokens': 0, 'latency_s': round(time.perf_counter() - started, 3)}, {})
        return response


def summarize(records):
    """Token/call totals by site and by turn, from runner records carrying `llm`."""
    by_site, by_turn, parts = {}, [], {}
    for record in records:
        calls = record.get('llm', [])
        turn = {'turn': record['turn'], 'replay': record.get('replay', False), 'kind': record.get('kind'),
                'calls': len(calls),
                'input_tokens': sum(c.get('input_tokens', 0) for c in calls),
                'output_tokens': sum(c.get('output_tokens', 0) for c in calls),
                'sites': {}}
        for c in calls:
            site = by_site.setdefault(c['site'], dict(calls=0, input_tokens=0, output_tokens=0,
                                                      latency_s=0.0, errors=0))
            site['calls'] += 1
            site['input_tokens'] += c.get('input_tokens', 0)
            site['output_tokens'] += c.get('output_tokens', 0)
            site['latency_s'] = round(site['latency_s'] + c.get('latency_s', 0), 3)
            site['errors'] += 'error' in c
            turn['sites'][c['site']] = turn['sites'].get(c['site'], 0) + 1
            total = sum(c.get('parts', {}).values())
            for name, size in c.get('parts', {}).items():
                # Estimated tokens: this part's character share of the measured input.
                share = parts.setdefault(c['site'], {}).setdefault(name, 0.0)
                parts[c['site']][name] = share + (c['input_tokens'] * size / total if total else 0)
        by_turn.append(turn)
    llm = {k: v for k, v in by_site.items() if k != 'embedding'}
    totals = {'llm_calls': sum(v['calls'] for v in llm.values()),
              'input_tokens': sum(v['input_tokens'] for v in llm.values()),
              'output_tokens': sum(v['output_tokens'] for v in llm.values()),
              'embedding_calls': by_site.get('embedding', {}).get('calls', 0),
              'embedding_tokens': by_site.get('embedding', {}).get('input_tokens', 0)}
    # Final S2 turn only: the replay and the separate first-request case come later.
    session = [t for t in by_turn if not t['replay'] and t['kind'] != 'first_request']
    return {'totals': totals, 'by_site': by_site, 'by_turn': by_turn,
            'estimated_input_parts': {s: {k: round(v) for k, v in p.items()} for s, p in parts.items()},
            'final_turn_input_tokens': session[-1]['input_tokens'] if session else 0,
            'method': 'Measured Bedrock usage per call; input parts estimated by character share.'}


def write_requests(requests, report_path):
    """Exact converse requests beside the report, for prompt-replay experiments."""
    from pathlib import Path
    path = Path(report_path).with_suffix('.requests.json')
    try:
        path.write_text(json.dumps(requests, default=str))
    except OSError as exc:
        print(f'Could not save replay requests to {path}: {exc}')


def print_summary(summary):
    t = summary['totals']
    print(f"\nModel calls: {t['llm_calls']}  input tokens: {t['input_tokens']}  output tokens: {t['output_tokens']}"
          f"  | embeddings: {t['embedding_calls']} calls, {t['embedding_tokens']} tokens")
    for site, v in sorted(summary['by_site'].items()):
        per = v['input_tokens'] // v['calls'] if v['calls'] else 0
        print(f"  {site:<10} calls={v['calls']:<3} in={v['input_tokens']:<7} out={v['output_tokens']:<6} "
              f"in/call={per:<6} latency={v['latency_s']:.1f}s errors={v['errors']}")
    print(f"  final-turn input tokens: {summary['final_turn_input_tokens']}")
