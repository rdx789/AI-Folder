---
name: add-tool
description: Scaffold a new agent tool in tools/ — one file with a TOOL_SPEC (Bedrock toolConfig shape) and a handle() function, then register it so the loop picks it up. Use whenever the agent needs a new tool.
---

# add-tool — scaffold one agent tool

Create a new tool for the agent from a short description (its name and what it
does).

## Steps

1. Pick a `snake_case` name for the tool.
2. Create `tools/<name>.py` with exactly two things:
   - `TOOL_SPEC` — a Bedrock `toolConfig` entry: a `toolSpec` with `name`,
     `description`, and an `inputSchema.json` JSON Schema. The schema MUST use
     `additionalProperties: false`, a `required` list, and a `description` on
     every field.
   - `handle(...)` — calls a small **mock receiver** function in the same file: a
     stand-in for the real backend that returns a simple, plausible mock output.
     No real services yet, so every tool runs on its mock.
3. Register the tool so the loop in `agent/` discovers it (follow whatever
   registration `tools/` already uses — e.g. add it to the registry/list).
4. One file, no business logic outside the handler.

## Output

State the tool name, its input fields, and confirm it's registered.
