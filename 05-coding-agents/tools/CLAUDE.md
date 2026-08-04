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
- For this personal-finance-analyst use case, build **at least 5 tools** the
  agent would actually need to answer questions against the `data/` dataset —
  e.g. list accounts, list/search transactions (by account, category,
  merchant, or date range), get spending by category, find recurring
  charges/subscriptions, get account balance, summarize spending over a
  period. Tools read from the JSON files in `data/` (loaded per the contract
  in `data/CLAUDE.md`) rather than a mock backend, since the dataset itself
  is the stand-in for a real backend.
- Every tool ends up with the same standardized shape (the skill guarantees it):
  - `TOOL_SPEC` — a `toolSpec` with `name`, `description`, and an
    `inputSchema.json` JSON Schema (`additionalProperties: false`, a `required`
    list, a `description` on every field).
  - `handle(...)` — reads and queries the JSON files in `data/` (per
    `data/CLAUDE.md`) and returns the result; the dataset is the stand-in for
    a real backend, so there's no separate mock receiver.
  - a test file under `tests/` that the skill runs until it's green.
- Tools are registered so the loop in `agent/` discovers them automatically.
