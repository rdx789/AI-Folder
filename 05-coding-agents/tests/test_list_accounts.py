from tools.list_accounts import handle


def test_happy_path_no_filter():
    result = handle()
    assert "accounts" in result
    assert len(result["accounts"]) == 20
    assert all("account_id" in a for a in result["accounts"])


def test_filter_by_account_type():
    result = handle(account_type="checking")
    assert all(a["account_type"] == "checking" for a in result["accounts"])
    assert len(result["accounts"]) > 0


def test_filter_with_no_matches_returns_empty():
    result = handle(account_type="bogus_type")
    assert result["accounts"] == []
