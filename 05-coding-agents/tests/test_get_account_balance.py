import pytest

from tools.get_account_balance import handle
from tools.list_accounts import handle as list_handle


def test_happy_path():
    acc_id = list_handle()["accounts"][0]["account_id"]
    result = handle(account_id=acc_id)
    assert result["account_id"] == acc_id
    assert "balance" in result
    assert "account_type" in result


def test_missing_required_field_raises():
    with pytest.raises(TypeError):
        handle()


def test_malformed_account_id_returns_error():
    result = handle(account_id="not-a-real-id")
    assert "error" in result
