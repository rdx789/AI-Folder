import pytest

from tools.check_refund_eligibility import handle


def test_happy_path_eligible():
    result = handle(order_id="ORD-98231", reason="damaged")
    assert result["found"] is True
    assert result["eligible"] is True


def test_happy_path_not_eligible_outside_window():
    result = handle(order_id="ORD-77104", reason="changed_mind")
    assert result["found"] is True
    assert result["eligible"] is False


def test_missing_order_id_raises():
    with pytest.raises(TypeError):
        handle(reason="damaged")


def test_missing_reason_raises():
    with pytest.raises(TypeError):
        handle(order_id="ORD-98231")


def test_invalid_reason_raises():
    with pytest.raises(ValueError):
        handle(order_id="ORD-98231", reason="buyers_remorse")


def test_unknown_order_id_not_found():
    result = handle(order_id="ORD-00000", reason="damaged")
    assert result == {"found": False}
