#!/usr/bin/env python3
"""Generate the personal-finance synthetic dataset per data/CLAUDE.md."""
import json
import os
import random
from datetime import date, timedelta

SEED = 42
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_START = date(2025, 1, 1)
WINDOW_END = date(2025, 12, 31)

CATEGORY_DEFS = [
    ("CAT-01", "income"),
    ("CAT-02", "transfer"),
    ("CAT-03", "subscriptions"),
    ("CAT-04", "groceries"),
    ("CAT-05", "dining"),
    ("CAT-06", "rent"),
    ("CAT-07", "utilities"),
    ("CAT-08", "transport"),
]
CAT_BY_NAME = {name: cid for cid, name in CATEGORY_DEFS}

MERCHANT_POOL = {
    "groceries": ["Trader Joe's", "Whole Foods", "Safeway", "Kroger", "Costco Wholesale"],
    "dining": ["Chipotle", "Olive Garden", "Starbucks", "Local Diner", "Sushi House"],
    "rent": ["Parkview Apartments", "Maple Street Realty", "Downtown Lofts"],
    "utilities": ["City Power & Light", "AquaFlow Water Co", "MetroGas Utility"],
    "transport": ["Uber", "Lyft", "Metro Transit Authority", "Shell Gas Station"],
    "subscriptions": ["Netflix", "Spotify", "Amazon Prime", "iCloud+", "Planet Fitness Membership"],
}

# name -> list of category names it may plausibly also be tagged as (small, named exceptions)
MERCHANT_EXCEPTIONS = {
    "Costco Wholesale": ["transport"],       # sells gas
    "Shell Gas Station": ["groceries"],      # convenience store snacks
    "Trader Joe's": ["dining"],              # prepared foods counter
    "Whole Foods": ["dining"],
    "Uber": ["dining"],                      # Uber Eats
}

SPEND_CATEGORY_WEIGHTS = {
    "groceries": 0.24,
    "dining": 0.28,
    "rent": 0.06,
    "utilities": 0.12,
    "transport": 0.20,
    "income": 0.06,
    "transfer": 0.04,
}

AMOUNT_RANGES = {
    "groceries": (-180.00, -20.00),
    "dining": (-70.00, -8.00),
    "rent": (-2200.00, -900.00),
    "utilities": (-250.00, -40.00),
    "transport": (-90.00, -5.00),
}

SUBSCRIPTION_AMOUNTS = {
    "Netflix": -15.99,
    "Spotify": -10.99,
    "Amazon Prime": -14.99,
    "iCloud+": -2.99,
    "Planet Fitness Membership": -24.99,
}

OWNER_NAMES = [
    "Alice Nguyen", "Brian Ortiz", "Carla Ibrahim", "David Kim", "Elena Petrova",
    "Felix Muller", "Grace Chen", "Hassan Ali", "Isabel Rossi", "Jamal Carter",
    "Katarina Novak", "Liam O'Brien", "Maya Patel", "Noah Levi", "Olivia Ferreira",
    "Priya Sharma", "Quinn Sullivan", "Rosa Delgado", "Samuel Osei", "Tara Whitfield",
]

ACCOUNT_TYPES = ["checking", "savings", "credit_card"]


def daterange_days(start, end):
    return (end - start).days


def random_date(rng, start, end):
    days = daterange_days(start, end)
    return start + timedelta(days=rng.randint(0, days))


def build_categories():
    return [{"category_id": cid, "name": name} for cid, name in CATEGORY_DEFS]


def build_merchants():
    merchants = []
    counter = 1
    recurring_ids = []
    exceptions_by_id = {}
    for cat_name, names in MERCHANT_POOL.items():
        for name in names:
            mid = f"MER-{counter:04d}"
            counter += 1
            merchants.append({
                "merchant_id": mid,
                "name": name,
                "default_category_id": CAT_BY_NAME[cat_name],
            })
            if cat_name == "subscriptions":
                recurring_ids.append(mid)
            if name in MERCHANT_EXCEPTIONS:
                exceptions_by_id[mid] = [CAT_BY_NAME[c] for c in MERCHANT_EXCEPTIONS[name]]
    assert len(merchants) == 25, f"expected 25 merchants, got {len(merchants)}"
    return merchants, recurring_ids, exceptions_by_id


def build_accounts(rng):
    accounts = []
    for i in range(1, 21):
        acc_id = f"ACC-{i:05d}"
        acc_type = rng.choices(ACCOUNT_TYPES, weights=[0.45, 0.30, 0.25])[0]
        opened_at = random_date(rng, date(2015, 1, 1), date(2024, 11, 1))
        if acc_type == "credit_card":
            balance = round(rng.uniform(-3200.00, 400.00), 2)
        else:
            balance = round(rng.uniform(150.00, 18000.00), 2)
        accounts.append({
            "account_id": acc_id,
            "owner_name": OWNER_NAMES[i - 1],
            "account_type": acc_type,
            "opened_at": opened_at.isoformat(),
            "balance": balance,
        })
    return accounts


def pick_tier(rng):
    roll = rng.random()
    if roll < 0.70:
        return "normal", rng.randint(15, 40)
    elif roll < 0.90:
        return "light", rng.randint(3, 10)
    else:
        return "heavy", rng.randint(60, 120)


def monthly_dates(rng, start, end):
    """Roughly one date per month between start and end, with jitter."""
    dates = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        day = min(28, cur.day) if cur.day else 1
        jitter = rng.randint(1, 27)
        try:
            d = date(cur.year, cur.month, jitter)
        except ValueError:
            d = date(cur.year, cur.month, 28)
        if start <= d <= end:
            dates.append(d)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return dates


def merchant_for_category(rng, cat_name, merchants_by_default_cat, exceptions_by_id, cat_by_name_for_id):
    # 85% chance: merchant whose default category matches; 15%: merchant that lists
    # this category as a named exception.
    if rng.random() < 0.85 or not any(
        cat_name in [cat_by_name_for_id[c] for c in cats] for cats in exceptions_by_id.values()
    ):
        pool = merchants_by_default_cat[cat_name]
        return rng.choice(pool)
    exception_candidates = [
        mid for mid, cats in exceptions_by_id.items()
        if cat_by_name_for_id.get(CAT_BY_NAME[cat_name]) and CAT_BY_NAME[cat_name] in cats
    ]
    if exception_candidates:
        return rng.choice(exception_candidates)
    return rng.choice(merchants_by_default_cat[cat_name])


def build_transactions(rng, accounts, merchants, recurring_ids, exceptions_by_id):
    merchant_by_id = {m["merchant_id"]: m for m in merchants}
    merchants_by_default_cat = {}
    for m in merchants:
        cat_name = next(name for name, cid in CAT_BY_NAME.items() if cid == m["default_category_id"])
        merchants_by_default_cat.setdefault(cat_name, []).append(m["merchant_id"])
    cat_id_by_name = CAT_BY_NAME
    cat_name_by_id = {cid: name for name, cid in CAT_BY_NAME.items()}

    transactions = []
    txn_counter = 1

    # Decide each account's tier first, then its subscription count, so a
    # merchant's ~12 yearly charges can never push a light account out of range.
    account_tiers = {acc["account_id"]: pick_tier(rng) for acc in accounts}

    SUBS_COUNT_BY_TIER = {
        "light": ([0], [1.0]),
        "normal": ([0, 1, 2], [0.4, 0.4, 0.2]),
        "heavy": ([1, 2, 3], [0.3, 0.4, 0.3]),
    }

    # Assign subscription merchants to a random subset of accounts (multiple
    # accounts per merchant, so each recurring merchant appears across accounts).
    subs_subscribers = {mid: [] for mid in recurring_ids}
    for acc in accounts:
        tier, _ = account_tiers[acc["account_id"]]
        choices, weights = SUBS_COUNT_BY_TIER[tier]
        n_subs = rng.choices(choices, weights=weights)[0]
        chosen = rng.sample(recurring_ids, k=min(n_subs, len(recurring_ids)))
        for mid in chosen:
            subs_subscribers[mid].append(acc["account_id"])

    subs_by_account = {}
    for mid, acc_ids in subs_subscribers.items():
        for acc_id in acc_ids:
            subs_by_account.setdefault(acc_id, []).append(mid)

    for acc in accounts:
        opened = date.fromisoformat(acc["opened_at"])
        window_start = max(WINDOW_START, opened)
        tier, target_count = account_tiers[acc["account_id"]]

        acc_transactions = []

        # Recurring subscription transactions first.
        for mid in subs_by_account.get(acc["account_id"], []):
            merchant = merchant_by_id[mid]
            base_amount = SUBSCRIPTION_AMOUNTS[merchant["name"]]
            for d in monthly_dates(rng, window_start, WINDOW_END):
                variance = round(rng.uniform(-0.30, 0.30), 2) if rng.random() < 0.2 else 0.0
                amount = round(base_amount + variance, 2)
                acc_transactions.append({
                    "transaction_id": None,
                    "account_id": acc["account_id"],
                    "merchant_id": mid,
                    "category_id": cat_id_by_name["subscriptions"],
                    "date": d.isoformat(),
                    "amount": amount,
                    "description": f"{merchant['name']} subscription",
                })

        remaining = max(0, target_count - len(acc_transactions))
        spend_cats = list(SPEND_CATEGORY_WEIGHTS.keys())
        weights = list(SPEND_CATEGORY_WEIGHTS.values())

        for _ in range(remaining):
            cat_name = rng.choices(spend_cats, weights=weights)[0]
            d = random_date(rng, window_start, WINDOW_END)

            if cat_name == "income":
                amount = round(rng.uniform(1000.00, 5200.00), 2)
                acc_transactions.append({
                    "transaction_id": None,
                    "account_id": acc["account_id"],
                    "merchant_id": None,
                    "category_id": cat_id_by_name["income"],
                    "date": d.isoformat(),
                    "amount": amount,
                    "description": "Payroll deposit",
                })
            elif cat_name == "transfer":
                is_in = rng.random() < 0.5
                amount = round(rng.uniform(50.00, 2000.00) * (1 if is_in else -1), 2)
                acc_transactions.append({
                    "transaction_id": None,
                    "account_id": acc["account_id"],
                    "merchant_id": None,
                    "category_id": cat_id_by_name["transfer"],
                    "date": d.isoformat(),
                    "amount": amount,
                    "description": "Transfer between accounts" if is_in else "Transfer to another account",
                })
            else:
                mid = merchant_for_category(
                    rng, cat_name, merchants_by_default_cat, exceptions_by_id, cat_name_by_id
                )
                merchant = merchant_by_id[mid]
                lo, hi = AMOUNT_RANGES[cat_name]
                amount = round(rng.uniform(lo, hi), 2)
                acc_transactions.append({
                    "transaction_id": None,
                    "account_id": acc["account_id"],
                    "merchant_id": mid,
                    "category_id": cat_id_by_name[cat_name],
                    "date": d.isoformat(),
                    "amount": amount,
                    "description": f"{merchant['name']} purchase",
                })

        for t in acc_transactions:
            t["transaction_id"] = f"TXN-{txn_counter:06d}"
            txn_counter += 1
            transactions.append(t)

    return transactions


def write_json(name, obj):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def main():
    rng = random.Random(SEED)

    categories = build_categories()
    merchants, recurring_ids, exceptions_by_id = build_merchants()
    accounts = build_accounts(rng)
    transactions = build_transactions(rng, accounts, merchants, recurring_ids, exceptions_by_id)

    write_json("categories.json", categories)
    write_json("merchants.json", merchants)
    write_json("accounts.json", accounts)
    write_json("transactions.json", transactions)

    print(f"categories: {len(categories)}")
    print(f"merchants: {len(merchants)}")
    print(f"accounts: {len(accounts)}")
    print(f"transactions: {len(transactions)}")


if __name__ == "__main__":
    main()