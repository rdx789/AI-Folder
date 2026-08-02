---
name: create-agent-tool
description: Create a production-quality agent tool end-to-end — its own file, a fully-described TOOL_SPEC, a working handle(), and a passing test suite — with no further prompting. Use whenever the agent needs a new tool; it owns the whole process from name to green tests.
---

# create-agent-tool — build one tool, end to end

Given a tool's name and what it does, produce a complete, tested, documented tool
with **no further prompting**. You own every step below; do not stop until the
tests are green.

## Process

1. **Scaffold.** Copy `templates/tool_template.py` to `tools/<name>.py` and fill
   it in. One tool per file.
2. **Schema.** Write the `inputSchema` as a real contract: `additionalProperties:
   false`, a `required` list, and a **thorough `description` on every field** —
   these are the model's instructions for each slot, so be specific and give
   allowed values or examples.
3. **Docstring.** The module docstring documents what the tool does, its inputs,
   and what it returns. A reader should understand the tool without running it.
4. **Handler + mock receiver.** Implement `handle(...)` so it calls a small
   **mock receiver** function in the same file — a stand-in for the real backend
   that returns a simple, plausible mock output. No real services yet, so every
   tool runs on its mock; the receiver is the seam where a real backend plugs in
   later.
5. **Tests.** Write `tests/test_<name>.py` covering: the happy path, each required
   field missing, and at least one malformed / out-of-range input.
6. **Try to break it.** Run
   `python .claude/skills/create-agent-tool/scripts/check_tool.py <name>` — it
   validates the TOOL_SPEC shape and runs the tests (under pytest; add pytest to
   requirements if missing). Fix whatever it flags and re-run until green.
7. **Register.** Add the tool so the `agent/` loop discovers it.

## Done means

`check_tool.py <name>` exits green and the tool is registered. Report the tool
name, its input fields, and the test results.
