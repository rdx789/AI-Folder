from tools.create_support_ticket import handle


def test_happy_path():
    result = handle(
        customer_email="sam@example.com",
        subject="Broken blender",
        description="Arrived broken, wants refund",
        priority="high",
        escalate=True,
    )
    assert result["ticket_id"]
    assert result["status"] == "escalated"


def test_missing_required_field():
    result = handle(
        customer_email="",
        subject="Broken blender",
        description="Arrived broken",
        priority=None,
        escalate=None,
    )
    assert "error" in result


def test_invalid_priority():
    result = handle(
        customer_email="sam@example.com",
        subject="Broken blender",
        description="Arrived broken",
        priority="super-urgent",
        escalate=None,
    )
    assert "error" in result
