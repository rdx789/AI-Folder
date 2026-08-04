from tools.get_spending_by_category import handle


def test_happy_path_no_filters():
    result = handle()
    assert len(result["spending_by_category"]) > 0
    assert all(row["total_spent"] <= 0 for row in result["spending_by_category"])


def test_filter_by_account_id():
    from tools.list_accounts import handle as list_handle
    acc_id = list_handle()["accounts"][0]["account_id"]
    result = handle(account_id=acc_id)
    assert isinstance(result["spending_by_category"], list)


def test_date_range_with_no_transactions_returns_empty():
    result = handle(start_date="1999-01-01", end_date="1999-01-02")
    assert result["spending_by_category"] == []


def test_sorted_biggest_spend_first():
    result = handle()
    totals = [row["total_spent"] for row in result["spending_by_category"]]
    assert totals == sorted(totals)
