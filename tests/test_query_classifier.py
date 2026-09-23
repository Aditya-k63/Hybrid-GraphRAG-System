from app.config import settings
from app.retrieval import query_classifier


def test_parse_valid():
    result = query_classifier._parse_json_response('{"type": "graph", "reason": "entities"}')
    assert result["type"] == "graph"
    assert result["retrieval_type"] == "graph"


def test_parse_with_fence():
    result = query_classifier._parse_json_response("```json\n{\"type\": \"vector\"}\n```")
    assert result["type"] == "vector"


def test_parse_unknown_type_defaults_to_hybrid():
    result = query_classifier._parse_json_response('{"type": "banana"}')
    assert result["type"] == "hybrid"


def test_parse_garbage_defaults_to_hybrid():
    result = query_classifier._parse_json_response("please classify this now")
    assert result["type"] == "hybrid"
    assert result["retrieval_type"] == "hybrid"


def test_classify_disabled_is_hybrid(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
    result = query_classifier.classify_query("Who founded Acme?")
    assert result["type"] == "hybrid"