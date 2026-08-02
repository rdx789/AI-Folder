# Comparison: this build vs. samples/04-create-agent-tool

Two passes at the same PROMPT.md task. The first pass here was a shortcut; the
second pass was brought up to parity with the reference sample. This records
what was wrong with the first pass and what changed.

## First pass (before this fix)

I invoked the `create-agent-tool` skill, then abandoned its process partway
through and hand-wrote a standalone `check_api_key.py` that made one bare
`converse()` call with no tools attached.

Gaps versus `samples/04-create-agent-tool`:

- **No tools.** `tools/CLAUDE.md` asks for at least 5 support tools built via
  the skill (order lookup, ticket status, escalation, refund eligibility,
  knowledge-base search). None existed — `tools/` was empty except for
  `CLAUDE.md`.
- **No tool-use loop.** `agent/CLAUDE.md` specifies a tool-use agent: send
  message + tool specs, execute any `toolUse` blocks, feed results back,
  repeat until `end_turn` or a turn cap. My script only ever proved the
  credentials worked — it never exercised tool-calling, so it wasn't
  actually testing "the agent," just "the API key."
- **No tool discovery.** The loop is supposed to auto-discover `tools/*.py`;
  there was no such mechanism.
- **Weaker `agent/client.py`.** Used `os.environ[...]` (bare `KeyError` on a
  missing var) instead of explicit `RuntimeError`s, and baked a single-turn,
  tool-less call into `converse()` instead of exposing a reusable
  `call_model(client, model_id, system, messages, tools)` that a loop could
  call with an evolving message list and tool specs.
- **Duplicate/parallel CLI.** `check_api_key.py` existed alongside what
  should have been the one real entrypoint, instead of `main.py` driving the
  actual agent.

Where it was already on par: the `prompts/` scenarios (5 sample messages +
system prompt + registry) matched the reference's shape and quality.

## Fix applied

Brought the project to parity with `samples/04-create-agent-tool`:

- **`agent/client.py`** rewritten to the seam pattern: `get_client()` /
  `get_model_id()` (explicit `RuntimeError` on missing env vars) /
  `call_model(client, model_id, system, messages, tools)` — a single Bedrock
  Converse call parameterized for reuse by a loop.
- **`agent/loop.py`** added: `_discover_tools()` scans `tools/*.py` for any
  module exporting `TOOL_SPEC` + `handle()` and registers it by name; `run()`
  drives the full turn — call model, execute `toolUse` blocks against the
  registry, append `toolResult` blocks, repeat up to `MAX_TURNS = 10`, return
  the final text.
- **5 tools added**, each via the `create-agent-tool` skill process (schema
  with `additionalProperties: false` + full field descriptions, docstring,
  mock receiver, `tests/test_<name>.py`), all green under
  `check_tool.py`:
  - `lookup_order` — order/account lookup by order_id and/or email
  - `check_ticket_status` — ticket status by ticket_id
  - `create_ticket` — open a new ticket
  - `check_refund_eligibility` — refund/return eligibility by order_id + reason
  - `search_knowledge_base` — keyword search over a help-center fixture set
- **`main.py`** replaces `check_api_key.py` as the single CLI entrypoint,
  now calling `agent.loop.run()` instead of a bare model call —
  `--list` / `--scenario NAME` / free-text message, same flags as the
  reference.
- **`__init__.py`** added to `agent/`, `prompts/`, `tools/`, `tests/` to make
  them real packages, matching the reference layout.
- Two sample prompts (`angry_refund_request`, `order_status_check`) updated
  to reference order IDs that exist in the `lookup_order` /
  `check_refund_eligibility` mock fixtures, so running them actually
  triggers a tool call instead of the model guessing.

## Verification

- `python .claude/skills/create-agent-tool/scripts/check_tool.py <name>`
  green for all 5 tools.
- `pytest tests -q` — 26 passed.
- `python main.py --scenario angry_refund_request` run live against Bedrock:
  the model called `check_refund_eligibility`, got back "3 days ago, 30-day
  window," and grounded its reply in that real tool output instead of
  fabricating a policy — confirming the discovery → call → tool-result →
  final-answer loop works end to end, not just the raw API call.

## Remaining differences from the reference (acceptable)

- Tool fixture data (order IDs, ticket IDs, KB articles) differs in specifics
  from the reference's, since both are independently authored mocks — same
  shape, different sample values.
- This project's `.env` uses `AWS_REGION` (not `AWS_DEFAULT_REGION`, which
  the reference sample's `.env.example` uses); `agent/client.py` here matches
  what's actually in this project's `.env`.
