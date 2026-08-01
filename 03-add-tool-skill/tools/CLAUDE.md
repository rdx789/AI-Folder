# tools/ — one tool per file

Every capability the agent has is a tool in this folder. **One tool per file.**

- **Create every tool with the `add-tool` skill** — don't hand-write tool files.
  Invoke the skill once per tool, telling it the tool's name and what it does; it
  scaffolds the file in the correct shape and registers it. If it doesn't trigger
  on its own, activate it explicitly.
- For this customer support use case, build **at least 5 tools** the support
  agent would actually need to handle the scenario — e.g. look up an order or
  account, check order / ticket status, search a help-center knowledge base,
  create or escalate a ticket, check refund or return eligibility, summarize the
  conversation.
- Every tool the skill produces has the same shape:
  - `TOOL_SPEC` — the Bedrock `toolConfig` entry: a `toolSpec` with `name`,
    `description`, and an `inputSchema.json` JSON Schema using
    `additionalProperties: false`, a `required` list, and a `description` on
    every field (the model reads those as instructions).
  - `handle(...)` — calls a small **mock receiver** in the same file: a stand-in
    for the real backend that returns a simple, plausible mock output (no real
    services yet, so every tool runs on its mock).
- Tools are registered so the loop in `agent/` discovers them automatically.
