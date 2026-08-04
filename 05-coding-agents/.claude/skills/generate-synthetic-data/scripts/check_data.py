#!/usr/bin/env python3
"""Validate data/ against the contract in data/CLAUDE.md."""
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")

errors = []
warnings = []


def load(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def check(cond, msg):
    if not cond:
        errors.append(msg)


def main():
    categories = load("categories.json")
    merchants = load("merchants.json")
    accounts = load("accounts.json")
    transactions = load("transactions.json")

    # --- row counts ---
    check(len(categories) == 8, f"expected 8 categories, got {len(categories)}")
    check(len(merchants) == 25, f"expected 25 merchants, got {len(merchants)}")
    check(len(accounts) == 20, f"expected 20 accounts, got {len(accounts)}")
    if not (400 <= len(transactions) <= 1000):
        warnings.append(f"transactions count {len(transactions)} is outside the ~800 ballpark (400-1000)")

    # --- formats & uniqueness ---
    cat_ids = [c["category_id"] for c in categories]
    check(len(set(cat_ids)) == len(cat_ids), "duplicate category_id")
    for c in categories:
        check(re.fullmatch(r"CAT-\d{2}", c["category_id"]), f"bad category_id format: {c['category_id']}")

    mer_ids = [m["merchant_id"] for m in merchants]
    check(len(set(mer_ids)) == len(mer_ids), "duplicate merchant_id")
    for m in merchants:
        check(re.fullmatch(r"MER-\d{4}", m["merchant_id"]), f"bad merchant_id format: {m['merchant_id']}")
        check(m["default_category_id"] in cat_ids, f"merchant {m['merchant_id']} default_category_id unresolved")

    acc_ids = [a["account_id"] for a in accounts]
    check(len(set(acc_ids)) == len(acc_ids), "duplicate account_id")
    for a in accounts:
        check(re.fullmatch(r"ACC-\d{5}", a["account_id"]), f"bad account_id format: {a['account_id']}")
        check(a["account_type"] in {"checking", "savings", "credit_card"}, f"bad account_type: {a['account_type']}")
        check(re.fullmatch(r"\d{4}-\d{2}-\d{2}", a["opened_at"]), f"bad opened_at format: {a['opened_at']}")
        if a["account_type"] != "credit_card":
            check(a["balance"] >= 0, f"non-credit_card account {a['account_id']} has negative balance")
        check(round(a["balance"], 2) == a["balance"], f"balance not 2dp: {a['account_id']}")

    acc_by_id = {a["account_id"]: a for a in accounts}
    cat_by_id = {c["category_id"]: c for c in categories}
    mer_by_id = {m["merchant_id"]: m for m in merchants}
    income_id = next(c["category_id"] for c in categories if c["name"] == "income")
    transfer_id = next(c["category_id"] for c in categories if c["name"] == "transfer")
    subs_id = next(c["category_id"] for c in categories if c["name"] == "subscriptions")

    txn_ids = [t["transaction_id"] for t in transactions]
    check(len(set(txn_ids)) == len(txn_ids), "duplicate transaction_id")

    window_start = date(2025, 1, 1)
    window_end = date(2025, 12, 31)

    per_account_count = defaultdict(int)
    per_merchant_accounts = defaultdict(set)
    subs_txns_by_merchant = defaultdict(list)

    for t in transactions:
        tid = t["transaction_id"]
        check(re.fullmatch(r"TXN-\d{6}", tid), f"bad transaction_id format: {tid}")
        check(t["account_id"] in acc_by_id, f"{tid}: account_id unresolved")
        check(t["category_id"] in cat_by_id, f"{tid}: category_id unresolved")
        check(round(t["amount"], 2) == t["amount"], f"{tid}: amount not 2dp")

        d = date.fromisoformat(t["date"])
        check(window_start <= d <= window_end, f"{tid}: date outside window")

        acc = acc_by_id.get(t["account_id"])
        if acc:
            opened = date.fromisoformat(acc["opened_at"])
            check(d >= opened, f"{tid}: date before account opened_at")
            per_account_count[t["account_id"]] += 1

        if t["category_id"] in (income_id, transfer_id):
            check(t["merchant_id"] is None, f"{tid}: income/transfer txn has merchant_id")
        else:
            check(t["merchant_id"] is not None, f"{tid}: non income/transfer txn missing merchant_id")
            check(t["merchant_id"] in mer_by_id, f"{tid}: merchant_id unresolved")

        if t["category_id"] == income_id:
            check(t["amount"] > 0, f"{tid}: income txn not positive")
        elif t["category_id"] != transfer_id:
            check(t["amount"] < 0, f"{tid}: non income/transfer txn not negative")

        if t["merchant_id"]:
            per_merchant_accounts[t["merchant_id"]].add(t["account_id"])
        if t["category_id"] == subs_id and t["merchant_id"]:
            subs_txns_by_merchant[t["merchant_id"]].append(t)

    # --- distribution: per-account tier membership ---
    tiers = {"light": (3, 10), "normal": (15, 40), "heavy": (60, 120)}
    for acc_id, count in per_account_count.items():
        ok = any(lo <= count <= hi for lo, hi in tiers.values())
        check(ok, f"account {acc_id} txn count {count} doesn't fit any tier range")

    # --- recurring subscriptions ---
    recurring_merchants = [
        mid for mid, txns in subs_txns_by_merchant.items()
        if len(per_merchant_accounts[mid]) >= 2 and len(txns) >= 2
    ]
    check(len(recurring_merchants) >= 2, "fewer than 2 recurring subscription merchants found across multiple accounts")

    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  - {e}")
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  - {w}")
        sys.exit(1)

    print("OK: all checks passed.")
    print(f"  categories={len(categories)} merchants={len(merchants)} accounts={len(accounts)} transactions={len(transactions)}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
