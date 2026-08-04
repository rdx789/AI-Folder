# Personal Finance Analyst Agent

A tool-use agent (Amazon Bedrock Converse) that answers personal-finance
questions by querying a synthetic accounts/transactions dataset — never by
guessing numbers.

See [`SUMMARY.md`](SUMMARY.md) for a full build writeup, and `PROMPT.md` for
the original spec.

## Layout

```
data/       synthetic dataset + generator + contract (data/CLAUDE.md)
agent/      Bedrock client seam + the tool-use loop (agent/CLAUDE.md)
prompts/    system prompt + selectable sample scenarios (prompts/CLAUDE.md)
tools/      one tool per file, reading directly from data/ (tools/CLAUDE.md)
tests/      one test file per tool
main.py     CLI entrypoint
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in AWS credentials, region, BEDROCK_MODEL_ID
```

Regenerate the dataset (only needed if `data/*.json` is missing or the
contract changes):

```bash
.venv/bin/python data/generate.py
```

## Usage

```bash
# list available scenarios
.venv/bin/python main.py list

# run one end to end (prints the tool-call trace + final answer)
.venv/bin/python main.py run recurring_subscriptions
```

## Tools

| Tool | Purpose |
|---|---|
| `list_accounts` | List accounts, optional `account_type` filter |
| `get_account_balance` | Balance + basic info for one account |
| `search_transactions` | Filter transactions by account/category/merchant/date/amount |
| `get_spending_by_category` | Spending totals by category, optional date range/account |
| `find_recurring_charges` | Detect same-merchant, ~monthly, similar-amount charges |

## Tests

```bash
.venv/bin/python -m pytest tests/
```
