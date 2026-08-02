from tools.search_knowledge_base import handle


def test_happy_path():
    result = handle(query="reset password", category="account", max_results=3)
    assert result["query"] == "reset password"
    assert isinstance(result["results"], list)
    assert all("url" in a for a in result["results"])


def test_missing_query():
    result = handle(query=None, category=None, max_results=None)
    assert "error" in result


def test_out_of_range_max_results():
    result = handle(query="billing", category=None, max_results=50)
    assert "error" in result
