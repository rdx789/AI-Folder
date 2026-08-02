# Customer support agent

A tool-use agent that answers customer support messages using Bedrock Converse
and a set of mock backend tools (order lookup, ticket status/creation, refund
eligibility, knowledge-base search). Useful both as a working support agent
demo and as a smoke test that the configured LLM API key, region, and model ID
are all wired up correctly.

## Setup

1. Copy the env example and fill in real values:

   ```
   cp .env.example .env
   ```

   Required variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_REGION`, `BEDROCK_MODEL_ID`.

2. Install dependencies:

   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

List the available sample scenarios:

```
python main.py --list
```

Run a specific scenario by name:

```
python main.py --scenario angry_refund_request
```

Send a custom message:

```
python main.py "I can't log in and my reset email never arrived."
```

The agent runs a full tool-use turn: it sends your message plus the tool specs
to the model, executes any tools the model calls (against mock backends),
feeds the results back, and prints the model's final reply. A successful run
confirms the credentials, region, and model ID all work end to end — and that
the tool-calling loop is functioning.

On failure (bad credentials, wrong region, invalid model ID, etc.) it prints
the error and exits non-zero.

## Project layout

- `agent/client.py` — the only file that imports boto3; `get_client()` /
  `get_model_id()` / `call_model()` are the provider seam.
- `agent/loop.py` — the tool-use loop: discovers tools in `tools/`, calls the
  model, executes any requested tool calls, feeds results back, repeats until
  a final answer or the turn cap.
- `tools/` — one file per tool (`lookup_order`, `check_ticket_status`,
  `create_ticket`, `check_refund_eligibility`, `search_knowledge_base`), each
  with a `TOOL_SPEC`, a `handle()`, and a mock backend. Dropping a new file
  here registers a new capability automatically.
- `prompts/` — the system prompt plus five sample customer-support scenarios,
  registered in `prompts/registry.py`.
- `tests/` — one test file per tool, covering the happy path, missing
  required fields, and malformed input.
- `main.py` — the CLI entrypoint.

## Adding a new scenario

Add a `prompts/<name>.py` file with `DESCRIPTION` and `PROMPT` module-level
strings, then register it in `prompts/registry.py`'s `SCENARIOS` dict.

## Adding a new tool

Use the `create-agent-tool` skill (`.claude/skills/create-agent-tool/`) — it
scaffolds the file, schema, mock receiver, and tests, then validates with
`python .claude/skills/create-agent-tool/scripts/check_tool.py <name>`.
