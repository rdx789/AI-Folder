import pytest

from tools.lookup_order import handle


def test_happy_path_by_order_id():
    result = handle(order_id="ORD-98231")
    assert result["found"] is True
    assert result["order_id"] == "ORD-98231"
    assert "status" in result
    assert "items" in result


def test_happy_path_by_email():
    result = handle(email="mlopez@example.com")
    assert result["found"] is True
    assert result["customer_email"] == "mlopez@example.com"


def test_missing_both_fields_raises():
    with pytest.raises(ValueError):
        handle()


def test_unknown_order_id_not_found():
    result = handle(order_id="ORD-00000")
    assert result == {"found": False}


def test_unknown_email_not_found():
    result = handle(email="nobody@example.com")
    assert result == {"found": False}
