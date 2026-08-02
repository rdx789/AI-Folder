import pytest

from tools.search_knowledge_base import handle


def test_happy_path_finds_relevant_article():
    result = handle(query="how do I reset my password")
    assert len(result["results"]) >= 1
    assert "password" in result["results"][0]["title"].lower()


def test_missing_query_raises():
    with pytest.raises(TypeError):
        handle()


def test_empty_query_raises():
    with pytest.raises(ValueError):
        handle(query="")


def test_max_results_out_of_range_raises():
    with pytest.raises(ValueError):
        handle(query="shipping", max_results=0)
    with pytest.raises(ValueError):
        handle(query="shipping", max_results=11)


def test_no_match_returns_empty_results():
    result = handle(query="xyzzy quux")
    assert result["results"] == []


def test_max_results_caps_output():
    result = handle(query="password reset shipping refund billing", max_results=2)
    assert len(result["results"]) <= 2
