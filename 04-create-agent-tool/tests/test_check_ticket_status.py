import pytest

from tools.check_ticket_status import handle


def test_happy_path_known_ticket():
    result = handle(ticket_id="TCK-4471")
    assert result["found"] is True
    assert result["ticket_id"] == "TCK-4471"
    assert result["status"] == "in_progress"


def test_missing_ticket_id_raises():
    with pytest.raises(TypeError):
        handle()


def test_empty_ticket_id_raises():
    with pytest.raises(ValueError):
        handle(ticket_id="")


def test_unknown_ticket_id_not_found():
    result = handle(ticket_id="TCK-0000")
    assert result == {"found": False}
