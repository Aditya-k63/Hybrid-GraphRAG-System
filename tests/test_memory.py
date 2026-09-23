import pytest

from app.config import settings
from app.memory import conversation
from app.memory.conversation import ConversationMemory, _lexical_score, _parse_keep_indices


def make_memory(histories: dict[str, list[dict]]) -> ConversationMemory:
    mem = ConversationMemory()
    for session_id, turns in histories.items():
        for role, content in turns:
            mem.add_message(session_id, role, content)
    return mem


class TestParseKeepIndices:
    def test_valid_json(self):
        assert _parse_keep_indices('{"keep": [0, 2, 5]}') == [0, 2, 5]

    def test_strips_fence_and_noise(self):
        assert _parse_keep_indices("```json\n{\"keep\": [1]}\n```") == [1]

    def test_rejects_garbage(self):
        assert _parse_keep_indices("no json here") is None
        assert _parse_keep_indices('{"keep": "nope"}') is None


class TestLexicalScore:
    def test_scores_overlap(self):
        assert _lexical_score("What does Acme do?", "Acme builds widgets.") > _lexical_score(
            "What does Acme do?", "Weather is nice today."
        )

    def test_stopwords_dont_count(self):
        assert _lexical_score("what is that", "what is that what is that") == 0


class TestTTL:
    def test_session_expires_after_ttl(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
        monkeypatch.setattr(settings, "MEMORY_TTL_SECONDS", 100)
        mem = make_memory({"s1": [("user", "alpha"), ("user", "beta")]})
        mem.last_seen["s1"] = 1000.0
        assert mem.select_relevant("s1", "alpha", now=1101.0) == []
        assert mem.get_history("s1") == []

    def test_session_alive_within_ttl(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
        monkeypatch.setattr(settings, "MEMORY_TTL_SECONDS", 100)
        mem = make_memory({"s1": [("user", "alpha")]})
        mem.last_seen["s1"] = 1000.0
        assert len(mem.select_relevant("s1", "alpha", now=1050.0)) == 1
        assert len(mem.get_history("s1")) == 1


class TestSelectRelevant:
    def test_empty_history(self):
        mem = make_memory({})
        assert mem.select_relevant("s1", "question") == []

    def test_fallback_uses_recent_when_no_overlap(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
        mem = make_memory({"s1": [("user", "unrelated chat"), ("user", "more talk")]})
        result = mem.select_relevant("s1", "about widgets", top_k=6)
        assert len(result) >= 1
        assert result[-1]["content"] == "more talk"

    def test_fallback_prefers_relevant_turn(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
        mem = make_memory({
            "s1": [
                ("user", "Let me explain how the Acme pipeline works."),
                ("user", "Nice weather today, right?"),
            ]
        })
        result = mem.select_relevant("s1", "Tell me about the Acme pipeline", top_k=1)
        assert len(result) == 1
        assert "Acme" in result[0]["content"]

    def test_caps_context_tokens(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
        long_turn = ("user", "The word " * 800)
        mem = make_memory({"s1": [long_turn, long_turn]})
        result = mem.select_relevant("s1", "anything", top_k=6, max_tokens=200)
        estimate = sum(len(m["content"]) for m in result) // 4
        assert estimate <= 200

    def test_llm_selector_rejected_indices_fall_back(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", True)
        mem = make_memory({"s1": [("user", "alpha"), ("user", "beta")]})
        monkeypatch.setattr(
            conversation,
            "Groq",
            lambda *args, **kwargs: _FakeGroq({"keep": [99, -1]}),
        )
        result = mem.select_relevant("s1", "alpha?", top_k=1)
        assert len(result) == 1

    def test_llm_selector_uses_kept_indices(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", True)
        mem = make_memory({"s1": [("user", "alpha"), ("user", "beta")]})
        monkeypatch.setattr(
            conversation,
            "Groq",
            lambda *args, **kwargs: _FakeGroq({"keep": [1]}),
        )
        result = mem.select_relevant("s1", "about beta", top_k=2)
        assert [m["content"] for m in result] == ["beta"]


class _FakeGroq:
    def __init__(self, payload: dict):
        self._payload = payload

    @property
    def chat(self):
        return self

    def completions(self):
        return self

    def create(self, **kwargs):
        return self

    @property
    def choices(self):
        return [self]

    @property
    def message(self):
        return self

    @property
    def content(self):
        import json
        return json.dumps(self._payload)