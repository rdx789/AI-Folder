---
name: generate-synthetic-data
description: Generate the personal-finance dataset in data/ (accounts, merchants, categories, transactions) from the contract in data/CLAUDE.md — a seeded, reproducible generator script plus validation. Use whenever the dataset needs to be (re)created or the contract in data/CLAUDE.md changes.
---

# generate-synthetic-data — build the data/ dataset from its contract

`data/CLAUDE.md` is the source of truth. Produce a generator script that satisfies
every rule in it, run it, and validate the output — with **no further prompting**.

## Process

1. **Read the contract.** Re-read `data/CLAUDE.md` in full before writing or
   editing anything — entity shapes, referential integrity, cross-field
   consistency, distributions, seed, row counts, date window, output format.
2. **Write the generator.** Create/update `data/generate.py`, a standalone
   Python script (stdlib only — `random`, `json`, `datetime`; no network, no
   extra dependencies) that:
   - seeds `random.seed(42)` before generating anything, and generates
     entities in a fixed order (`categories` → `merchants` → `accounts` →
     `transactions`) so the run is byte-identical across runs of the same
     script version;
   - builds `categories`, `merchants`, `accounts`, then `transactions` in that
     order, satisfying every referential-integrity and cross-field rule as it
     goes (don't generate first and patch violations after);
   - implements the account-mix distribution (70% normal / 20% light / 10%
     heavy transaction counts) and the recurring-subscription merchants;
   - writes each entity as pretty-printed (`indent=2`) JSON to
     `data/accounts.json`, `data/merchants.json`, `data/categories.json`,
     `data/transactions.json`.
3. **Run it.** Execute `python data/generate.py` from the project root and
   confirm it exits cleanly and writes all four files.
4. **Validate.** Run
   `python .claude/skills/generate-synthetic-data/scripts/check_data.py` — it
   checks row counts, formats, referential integrity, cross-field rules, and
   the distribution/recurring-merchant requirements from the contract. Fix
   whatever it flags in `data/generate.py`, regenerate, and re-run until
   green.
5. **Reproducibility check.** Run the generator a second time and confirm the
   output files are byte-identical to the first run (same seed, same script
   → same bytes).

## Done means

`check_data.py` exits green, a second run of `data/generate.py` produces
byte-identical output, and every file in `data/` matches the contract in
`data/CLAUDE.md`. Report the row counts per entity and the validation result.