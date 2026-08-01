# Customer Support Agent CLI

A small command-line tool that tests whether an LLM API key works for a
customer support use case. It sends a representative support scenario to a
Bedrock model through a tool-use loop and prints the model's answer.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in real values:

   ```bash
   cp .env.example .env
   ```

   - `BEDROCK_MODEL_ID` — the Bedrock model ID (or cross-region inference
     profile ID) to call.
   - `AWS_REGION` — the AWS region for Bedrock, e.g. `us-east-1`.

   Standard AWS credentials (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`,
   or a configured profile/role) must also be available in the environment —
   `boto3` picks these up automatically.

## Usage

List the available support scenarios:

```bash
python main.py --list
```

Run one:

```bash
python main.py order-status-check
```

This sends the scenario's sample prompt to the model, lets it call any of
the 5 support tools (order lookup, ticket status, knowledge-base search,
ticket creation, refund eligibility) as needed, and prints the final answer.

## Project layout

- `agent/` — the model-client seam (`agent/client.py`) and the tool-use loop
  (`agent/loop.py`).
- `tools/` — one file per tool; each exposes a `TOOL_SPEC` and `handle()`
  backed by a mock receiver. Auto-discovered by `tools/registry.py`.
- `prompts/` — the system prompt and 5 sample support scenarios, registered
  in `prompts/scenarios.py`.
- `main.py` — the CLI entry point.
