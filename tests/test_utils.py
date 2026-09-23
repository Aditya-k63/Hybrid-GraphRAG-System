import pytest

from app.utils import retry as retry_module
from app.utils.json_utils import extract_json_object
from app.utils.retry import CircuitBreaker, with_retry
from app.utils.text import chars_for_tokens, estimate_tokens


class _RateLimitedError(Exception):
    def __init__(self, status_code, message="rate limited"):
        super().__init__(message)
        self.status_code = status_code


class TestWithRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(retry_module.time, "sleep", lambda seconds: None)
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=1.0)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _RateLimitedError(429)
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_does_not_retry_non_rate_limit_errors(self, monkeypatch):
        monkeypatch.setattr(retry_module.time, "sleep", lambda seconds: None)
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=1.0)
        def boom():
            calls["n"] += 1
            raise ValueError("bad output")

        with pytest.raises(ValueError):
            boom()
        assert calls["n"] == 1

    def test_raises_after_exhaustion(self, monkeypatch):
        monkeypatch.setattr(retry_module.time, "sleep", lambda seconds: None)
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=1.0)
        def always_limited():
            calls["n"] += 1
            raise _RateLimitedError(429)

        with pytest.raises(_RateLimitedError):
            always_limited()
        assert calls["n"] == 3


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(threshold=2, cooldown_seconds=60)
        assert not breaker.is_open()
        breaker.record_failure()
        assert not breaker.is_open()
        breaker.record_failure()
        assert breaker.is_open()

    def test_success_resets(self):
        breaker = CircuitBreaker(threshold=2, cooldown_seconds=60)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open()


class TestText:
    def test_estimate_tokens(self):
        assert estimate_tokens("hello world") == 2
        assert estimate_tokens("") == 0

    def test_chars_for_tokens(self):
        assert chars_for_tokens(150) == 600


class TestJsonUtils:
    def test_extracts_object_with_fence(self):
        assert extract_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}

    def test_raises_on_garbage(self):
        with pytest.raises(ValueError):
            extract_json_object("no json here")

    def test_raises_on_non_object(self):
        with pytest.raises(ValueError):
            extract_json_object("[1, 2, 3]")