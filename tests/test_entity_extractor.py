import pytest

from app.ingestion import entity_extractor
from app.ingestion.entity_extractor import (
    ExtractionCircuitBreaker,
    _clean_entities,
    _clean_relationships,
    _extract_json_object,
    extract_entities_llm,
)


class TestJsonExtraction:
    def test_strips_code_fences(self):
        raw = "```json\n{\"entities\": []}\n```"
        assert _extract_json_object(raw) == {"entities": []}

    def test_extracts_nested_object(self):
        raw = "Sure, here you go: {\"entities\": [{\"name\": \"Acme\"}]}. Done."
        result = _extract_json_object(raw)
        assert result["entities"][0]["name"] == "Acme"

    def test_rejects_non_object(self):
        with pytest.raises(ValueError):
            _extract_json_object("just some text")

    def test_rejects_invalid_json(self):
        with pytest.raises(ValueError):
            _extract_json_object("{invalid}")


class TestCleaning:
    def test_dedup_and_type_normalize(self):
        raw = [
            {"name": "Acme", "type": "ORGANIZATION", "description": "a firm"},
            {"name": "acme", "type": "UNKNOWN", "description": "dup"},
            {"name": "  ", "type": "PERSON", "description": "blank name"},
        ]
        result = _clean_entities(raw)
        assert len(result) == 1
        assert result[0]["type"] == "ORGANIZATION"

    def test_unknown_type_defaults_to_concept(self):
        (result,) = _clean_entities([{"name": "Widget", "type": "NOT_A_TYPE", "description": "thing"}])
        assert result["type"] == "CONCEPT"

    def test_drops_missing_description_if_required(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "ENTITY_REQUIRE_DESCRIPTION", True)
        assert _clean_entities([{"name": "Ghost", "type": "PERSON"}]) == []

        monkeypatch.setattr(settings, "ENTITY_REQUIRE_DESCRIPTION", False)
        (result,) = _clean_entities([{"name": "Ghost", "type": "PERSON"}])
        assert result["description"] == "Extracted entity: Ghost"

    def test_relationships_filter_dangling_endpoints(self):
        entities = [{"name": "Groq", "type": "ORGANIZATION", "description": "an llm"}]
        names = {entity["name"].lower() for entity in entities}
        raw = [
            {"source": "Groq", "target": "PyTorch", "relation": "competes_with"},
            {"source": "Groq", "target": "", "relation": "bad"},
            {"source": "Nope", "target": "Groq", "relation": "bad"},
        ]
        assert _clean_relationships(raw, names) == []

        raw.append({"source": "Groq", "target": "Groq", "relation": "is"})
        assert _clean_relationships(raw, names) == [{"source": "Groq", "target": "Groq", "relation": "is"}]


class TestCircuitBreaker:
    def test_opens_after_threshold(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "ENTITY_CIRCUIT_BREAKER_THRESHOLD", 3)
        monkeypatch.setattr(settings, "ENTITY_CIRCUIT_BREAKER_COOLDOWN", 60)
        breaker = ExtractionCircuitBreaker()
        for _ in range(settings.ENTITY_CIRCUIT_BREAKER_THRESHOLD):
            assert not breaker.is_open()
            breaker.record_failure()
        assert breaker.is_open()

    def test_success_resets(self):
        breaker = ExtractionCircuitBreaker()
        breaker.record_failure()
        breaker.record_success()
        assert not breaker.is_open()


def test_extract_entities_llm_disabled(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
    assert extract_entities_llm("any text") == {"entities": [], "relationships": []}