# Comparison against samples/05-schema-subagent

Reference: `../../samples/05-schema-subagent`. Both implementations follow the
same architecture required by the CLAUDE.md files: boto3 confined to
`agent/client.py`, one Bedrock Converse call, `pkgutil`-based tool
auto-discovery, one tool per file (5 tools), one prompt per file (5
scenarios + system prompt + registry), argparse CLI.

## Fixes adopted from the reference

1. **Native JSON tool results.** The reference returns `{"json": output}` in
   `toolResult.content` instead of `{"text": json.dumps(output)}`. Converse
   supports typed JSON content directly — adopted in `agent/loop.py`.
2. **Exception guard around handlers.** The reference wraps
   `handler(**tool_use["input"])` in try/except so a buggy tool returns a
   `toolResult` error instead of crashing the whole loop. Adopted in
   `agent/loop.py`.

## Differences kept as-is (not fixes, judgment calls)

- **Tool discovery location.** This project keeps it in
  `agent/tool_registry.py`; the reference inlines `_discover_tools()` in
  `loop.py`. Equivalent behavior, minor separation-of-concerns preference.
- **System prompt wiring.** This project passes `system_prompt` into
  `run_agent(...)` as a parameter, keeping `agent/` decoupled from
  `prompts/`. The reference imports `SYSTEM_PROMPT` directly inside
  `loop.py`, creating a one-way `agent/` → `prompts/` dependency.
  `prompts/CLAUDE.md` says the loop and tool-calling live in `agent/`; this
  project's version avoids `agent/` depending on `prompts/` at all.
- **`lookup_order` required fields.** This project requires `order_id`. The
  reference allows either `order_id` or `customer_email` alone
  (`"required": []`). Different business assumption about what a lookup
  needs; neither is more spec-compliant.
- **Tool docstrings.** The reference's tool files have no module docstring.
  This project keeps the module docstring the `create-agent-tool` skill
  instructions call for ("a reader should understand the tool without
  running it").
- **`--message` free-text flag.** The reference's CLI also accepts an
  arbitrary `--message` in addition to `--scenario`. Not added here — out of
  scope for what `prompts/CLAUDE.md` asks for (a registry of named
  scenarios), but worth considering as a follow-up if free-text testing is
  wanted.
