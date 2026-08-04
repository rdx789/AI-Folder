from tools.search_transactions import handle


def test_happy_path_no_filter():
    result = handle()
    assert result["count"] == len(result["transactions"])
    assert result["count"] > 0


def test_filter_by_account_id():
    txns = handle()["transactions"]
    acc_id = txns[0]["account_id"]
    result = handle(account_id=acc_id)
    assert all(t["account_id"] == acc_id for t in result["transactions"])
    assert result["count"] > 0


def test_filter_by_date_range():
    result = handle(start_date="2025-06-01", end_date="2025-06-30")
    assert all("2025-06-01" <= t["date"] <= "2025-06-30" for t in result["transactions"])


def test_filter_by_amount_range():
    result = handle(min_amount=-50, max_amount=0)
    assert all(-50 <= t["amount"] <= 0 for t in result["transactions"])


def test_no_matches_returns_empty():
    result = handle(account_id="ACC-99999")
    assert result["transactions"] == []
    assert result["count"] == 0
