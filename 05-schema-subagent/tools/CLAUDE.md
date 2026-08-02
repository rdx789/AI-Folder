# tools/ — one tool per file

Every capability the agent has is a tool in this folder. **One tool per file.**

- **Create every tool with the `create-agent-tool` skill** — don't hand-write
  tool files. Invoke it once per tool with the tool's name and what it does; it
  owns the whole job: the file, a fully-described schema, a docstring, a passing
  test suite, and registration. If it doesn't trigger on its own, activate it
  explicitly.
- The skill delegates **input-schema design to the `schema-smith` subagent** — a
  focused, isolated context that returns a finished, validated JSON Schema. Let
  it; don't hand-roll schemas.
- For this customer support use case, build **at least 5 tools** the support
  agent would actually need — e.g. look up an order or account, check order /
  ticket status, search a help-center knowledge base, create or escalate a
  ticket, check refund or return eligibility, summarize the conversation.
- Every tool ends up with the same standardized shape (the skill guarantees it):
  - `TOOL_SPEC` — a `toolSpec` with `name`, `description`, and an
    `inputSchema.json` JSON Schema (`additionalProperties: false`, a `required`
    list, a `description` on every field).
  - `handle(...)` — calls a **mock receiver** in the same file: a stand-in for the
    real backend returning a simple mock output (no real services yet).
  - a test file under `tests/` that the skill runs until it's green.
- Tools are registered so the loop in `agent/` discovers them automatically.
