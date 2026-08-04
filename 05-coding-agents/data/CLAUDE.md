# CLAUDE.md — data contract for the personal-finance dataset

This file is the source of truth for `data/`. The `generate-synthetic-data`
skill must produce output that satisfies every rule below, and the tools must
assume every rule below holds — no defensive re-validation of the data itself.

## Entities

### `accounts` (`data/accounts.json`)
- `account_id`: string, format `ACC-#####` (5 digits, zero-padded), unique.
- `owner_name`: string.
- `account_type`: enum `{checking, savings, credit_card}`.
- `opened_at`: date (`YYYY-MM-DD`).
- `balance`: number, 2 decimal places. Negative allowed only for `credit_card`.

### `merchants` (`data/merchants.json`)
- `merchant_id`: string, format `MER-####`, unique.
- `name`: string.
- `default_category_id`: string, must resolve to a `categories.category_id`.

### `categories` (`data/categories.json`)
- `category_id`: string, format `CAT-##`, unique.
- `name`: string, e.g. `groceries`, `dining`, `rent`, `utilities`, `entertainment`,
  `transport`, `income`, `transfer`, `subscriptions`, `other`.

### `transactions` (`data/transactions.json`)
- `transaction_id`: string, format `TXN-######`, unique.
- `account_id`: must resolve to an `accounts.account_id`.
- `merchant_id`: must resolve to a `merchants.merchant_id`, except when
  `category_id` is `income` or `transfer`, where it may be `null`.
- `category_id`: must resolve to a `categories.category_id`.
- `date`: date (`YYYY-MM-DD`), within the dataset's fixed window (see below).
- `amount`: number, 2 decimal places. Negative = money out, positive = money in.
  `income`/`transfer`-in rows are positive; everything else is negative.
- `description`: string.

## Referential integrity

- Every `merchant_id`, `category_id`, and `account_id` referenced by a
  `transactions` row resolves to an existing row in the corresponding entity.
- Every `merchants.default_category_id` resolves to an existing `categories.category_id`.
- Every account has at least one transaction; every category used by a
  merchant is a real category.

## Cross-field consistency

- Every transaction's `date` must be on or after its account's `opened_at`.
- `credit_card` accounts may carry a negative `balance`; `checking`/`savings`
  balances stay non-negative.
- A transaction's `category_id` is plausible for its `merchant_id`: either it
  matches the merchant's `default_category_id`, or it's drawn from a small,
  named set of allowed exceptions per merchant — not an arbitrary category.
- `income` and `transfer` transactions never carry a `merchant_id`.

## Realistic distributions

- Transaction volume per account is not uniform: roughly 70% of accounts get
  a "normal" transaction count (15–40 over the window), 20% get "light" use
  (3–10), 10% are "heavy" users (60–120) — the long tail that makes aggregate
  queries ("top spender") meaningful.
- At least 2 merchants per category recur across multiple accounts (so
  `recurring_charges`-style tools have something to find) — flag these as
  `subscriptions`-category transactions: same merchant, ~monthly cadence,
  same or near-identical amount.
- Spend amounts follow category-appropriate ranges, not one flat
  distribution (e.g. `rent` is large and near-monthly; `dining` is small and
  frequent).

## Reproducibility & size

- Fixed random seed: `42`. Same seed + same script version → byte-identical
  output.
- Row counts: 20 `accounts`, 8 `categories`, 25 `merchants`, ~800
  `transactions` total (driven by the distribution above, not a flat
  per-account count).
- Fixed date window: `2025-01-01` through `2025-12-31`.

## Output format

- JSON, one file per entity, array of flat objects, written to `data/`.
- Pretty-printed (indent 2) for readability during development.