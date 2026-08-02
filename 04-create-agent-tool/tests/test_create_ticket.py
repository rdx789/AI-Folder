import pytest

from tools.create_ticket import handle


def test_happy_path():
    result = handle(
        subject="Damaged item",
        description="Blender arrived cracked, order ORD-98231",
        priority="high",
        customer_email="jsmith@example.com",
    )
    assert result["status"] == "open"
    assert result["subject"] == "Damaged item"
    assert result["priority"] == "high"
    assert result["ticket_id"].startswith("TCK-")


def test_missing_subject_raises():
    with pytest.raises(TypeError):
        handle(description="x", priority="low", customer_email="a@example.com")


def test_missing_description_raises():
    with pytest.raises(TypeError):
        handle(subject="x", priority="low", customer_email="a@example.com")


def test_invalid_priority_raises():
    with pytest.raises(ValueError):
        handle(subject="x", description="y", priority="critical", customer_email="a@example.com")


def test_empty_customer_email_raises():
    with pytest.raises(ValueError):
        handle(subject="x", description="y", priority="low", customer_email="")
