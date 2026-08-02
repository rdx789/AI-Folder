from tools.check_ticket_status import handle


def test_happy_path():
    result = handle(ticket_id="T-5521")
    assert result["ticket_id"] == "T-5521"
    assert "status" in result
    assert "priority" in result
    assert "subject" in result
    assert "updated_at" in result


def test_missing_ticket_id():
    result = handle(ticket_id=None)
    assert "error" in result


def test_malformed_ticket_id():
    result = handle(ticket_id="not_a_valid_id")
    assert "error" in result
