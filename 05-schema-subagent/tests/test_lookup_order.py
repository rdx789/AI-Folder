from tools.lookup_order import handle


def test_happy_path():
    result = handle(order_id="A1029", customer_email="sam@example.com")
    assert result["order_id"] == "A1029"
    assert "status" in result
    assert "items" in result
    assert "total" in result
    assert "shipping" in result


def test_missing_order_id():
    result = handle(order_id=None, customer_email="sam@example.com")
    assert "error" in result


def test_malformed_order_id():
    result = handle(order_id="!!!", customer_email=None)
    assert "error" in result
