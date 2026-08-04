from tools.find_recurring_charges import handle


def test_happy_path_finds_recurring_charges():
    result = handle()
    assert len(result["recurring_charges"]) >= 2
    for row in result["recurring_charges"]:
        assert row["occurrences"] >= 3


def test_min_occurrences_filter_raises_bar():
    result = handle(min_occurrences=100)
    assert result["recurring_charges"] == []


def test_scoped_to_account_with_no_recurring_returns_empty_or_subset():
    all_result = handle()
    if all_result["recurring_charges"]:
        acc_id = all_result["recurring_charges"][0]["account_id"]
        scoped = handle(account_id=acc_id)
        assert all(row["account_id"] == acc_id for row in scoped["recurring_charges"])


def test_no_matching_account_returns_empty():
    result = handle(account_id="ACC-99999")
    assert result["recurring_charges"] == []
