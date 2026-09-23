import logging
import threading
import time

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = (429, 529)


def _is_retryable(error: Exception, statuses: tuple[int, ...]) -> bool:
    status = getattr(error, "status_code", None)
    if status in statuses:
        return True
    message = str(error)
    return any(str(code) in message for code in statuses)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    backoff: float = 2.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
):
    """Mirror Claude Code's withRetry.ts: bounded attempts with exponential backoff
    on rate-limit style errors. Raises the last error once attempts are exhausted."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = max(1, max_attempts)
            last_error: Exception | None = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:
                    last_error = exc
                    exhausted = attempt >= attempts - 1
                    if exhausted or not _is_retryable(exc, _RETRYABLE_STATUSES):
                        raise
                    wait = base_delay * (backoff**attempt)
                    logger.warning(
                        "%s retrying in %ss (attempt %d/%d)",
                        func.__name__,
                        wait,
                        attempt + 1,
                        attempts,
                    )
                    time.sleep(wait)
            raise RuntimeError(f"with_retry exhausted unexpectedly for {func.__name__}") from last_error

        return wrapper

    return decorator


class CircuitBreaker:
    """Opens after N consecutive failures and stays open for a cooldown window.
    record_success resets the failure count."""

    def __init__(
        self,
        threshold: int = 3,
        cooldown_seconds: float = 300.0,
        name: str = "circuit_breaker",
    ):
        self.threshold = max(1, threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.name = name
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        with self._lock:
            if time.monotonic() < self._open_until:
                return True
            if self._open_until and time.monotonic() >= self._open_until:
                self._open_until = 0.0
                self._consecutive_failures = 0
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.threshold:
                self._open_until = time.monotonic() + self.cooldown_seconds
                logger.warning(
                    "%s opened for %.1fs after %d consecutive failures",
                    self.name,
                    self.cooldown_seconds,
                    self._consecutive_failures,
                )