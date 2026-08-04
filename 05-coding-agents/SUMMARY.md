# Personal Finance Analyst Agent — Build Summary

Built per `PROMPT.md` and the CLAUDE.md contracts in `data/`, `agent/`, `prompts/`, `tools/`.

## 1. Data (`data/`)

- `generate.py` — stdlib-only generator, seed 42, fixed build order
  (categories → merchants → accounts → transactions). Byte-reproducible across runs.
- Output: 8 categories, 25 merchants, 20 accounts, 438 transactions, all in
  `data/*.json`.
- `.claude/skills/generate-synthetic-data/scripts/check_data.py` — didn't exist,
  so it was written to validate row counts, ID formats, referential integrity,
  cross-field rules (date ≥ opened_at, credit_card-only negative balances,
  income/transfer never carry a merchant_id), the 70/20/10 transaction-volume
  tiers, and ≥2 recurring subscription merchants spanning multiple accounts.
  **Green.**

## 2. Agent core (`agent/`)

- `client.py` — the only file importing boto3; `get_client()` / `get_model_id()`
  read region/model from env (`.env` via `load_dotenv(find_dotenv())`).
- `registry.py` — auto-discovers every module under `tools/` via `pkgutil`;
  no tool is named or imported by hand.
- `loop.py` — the Converse tool-use loop: send request + tool specs, run any
  `toolUse` the model returns, feed back `toolResult`, repeat, capped at
  `MAX_TURNS = 8`. Per-tool exceptions are caught and returned as a
  `toolResult` error (not a crash).

## 3. Prompts (`prompts/`)

- `system_prompt.py` — frames the model as a personal finance analyst that
  must use tools rather than invent numbers.
- 5 sample scenarios, one per file: `dining_last_month.py`,
  `recurring_subscriptions.py`, `most_active_account.py`,
  `groceries_quarter_comparison.py`, `unusual_large_transactions.py`.
- `registry.py` — `list_scenarios()` / `get_scenario(name)`, so scenarios are
  discoverable and selectable without touching the loop.

## 4. Tools (`tools/`)

5 tools, each with a `TOOL_SPEC` (schema-designed by a `schema-smith` subagent
call), a docstring, and `handle(...)` reading directly from `data/*.json`
(per `tools/CLAUDE.md`, no mock backend — the dataset is the stand-in):

| Tool | Purpose |
|---|---|
| `list_accounts` | List accounts, optional `account_type` filter |
| `get_account_balance` | Balance + basic info for one `account_id` |
| `search_transactions` | Filter transactions by account/category/merchant/date/amount |
| `get_spending_by_category` | Totals of money out, grouped by category, optional date range/account |
| `find_recurring_charges` | Detects same-merchant, ~monthly, similar-amount charges |

Tests under `tests/` (19 total: happy path, missing required fields, malformed/
out-of-range inputs). All pass via
`.claude/skills/create-agent-tool/scripts/check_tool.py <name>`.

## 5. Wiring

- `main.py` — CLI: `list` (show scenarios), `run <scenario>` (execute one
  end to end, printing the tool-call trace and final answer).
- `requirements.txt` — boto3, python-dotenv, pytest.
- `.venv` — created because the system Python is externally-managed (PEP 668).

## 6. Environment fix

`.env`'s `BEDROCK_MODEL_ID` was `amazon.titan-embed-text-v2:0` (embeddings-only,
can't do Converse/tool-use). Per your instruction, changed to
`us.amazon.nova-2-lite-v1:0`. Verified with a live Converse call before
building the rest.

## 7. Live proof

- `main.py run recurring_subscriptions` → one `find_recurring_charges({})`
  call; correctly surfaced Netflix/Spotify/Planet Fitness/iCloud+/Amazon Prime
  across multiple accounts; final answer totaled ~$164.79/month.
- `main.py run most_active_account` → `list_accounts` + one
  `search_transactions` call per account; correctly identified `ACC-00005`
  with 27 transactions as the most active account.

Both answers are only derivable by querying `data/`, not guessable from the
prompt alone.
