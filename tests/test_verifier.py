import pytest

from app.config import settings
from app.generation.verifier import verify_answer


class _Completions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return _Response(self._content)


class _Response:
    def __init__(self, content: str):
        self._content = content

    @property
    def choices(self):
        return [_Choice(self._content)]


class _Choice:
    def __init__(self, content: str):
        self._content = content

    @property
    def message(self):
        return _Message(self._content)


class _Message:
    def __init__(self, content: str):
        self._content = content

    @property
    def content(self):
        return self._content


class _Chat:
    def __init__(self, content: str):
        self.completions = _Completions(content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _Chat(content)


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", False)
    monkeypatch.setattr(settings, "ANSWER_VERIFICATION_ENABLED", True)
    assert verify_answer("q", [], "a") is None


def test_feature_flag_off_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", True)
    monkeypatch.setattr(settings, "ANSWER_VERIFICATION_ENABLED", False)
    assert verify_answer("q", [], "a") is None


def test_valid_output_returned(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", True)
    monkeypatch.setattr(settings, "ANSWER_VERIFICATION_ENABLED", True)
    client = _FakeClient('{"grounded": true, "confidence": 0.9, "note": "cites context"}')
    result = verify_answer("q", [{"content": "facts"}], "a", client=client)
    assert result == {"grounded": True, "confidence": 0.9, "note": "cites context"}


def test_garbage_output_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", True)
    monkeypatch.setattr(settings, "ANSWER_VERIFICATION_ENABLED", True)
    client = _FakeClient("not json at all")
    assert verify_answer("q", [{"content": "facts"}], "a", client=client) is None


def test_invalid_confidence_rejected(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_LIVE_LLM", True)
    monkeypatch.setattr(settings, "ANSWER_VERIFICATION_ENABLED", True)
    client = _FakeClient('{"grounded": true, "confidence": 5.0}')
    assert verify_answer("q", [{"content": "facts"}], "a", client=client) is None