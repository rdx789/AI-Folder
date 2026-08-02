from tools.check_refund_eligibility import handle


def test_happy_path():
    result = handle(order_id="A1029", item_sku=None, reason="defective")
    assert result["order_id"] == "A1029"
    assert result["eligible"] is True


def test_missing_order_id():
    result = handle(order_id=None, item_sku=None, reason=None)
    assert "error" in result


def test_invalid_reason():
    result = handle(order_id="A1029", item_sku=None, reason="because")
    assert "error" in result
