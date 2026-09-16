import time
import hashlib
import hmac
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if not key or not hmac.compare_digest(key, settings.API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it as header: X-API-Key",
        )
    return key


def check_api_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    if not key or not hmac.compare_digest(key, settings.API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it as header: X-API-Key",
        )
    return key


class RateLimiter:
    def __init__(self, max_requests: int = None, window: int = None):
        self.max_requests = max_requests or settings.RATE_LIMIT_REQUESTS
        self.window = window or settings.RATE_LIMIT_WINDOW
        self.requests: dict[str, list[float]] = {}
        self._ci_bypass = os.getenv("CI") == "true" or os.getenv("RATE_LIMIT_BYPASS") == "true"

    def is_allowed(self, key: str) -> bool:
        if self._ci_bypass:
            return True

        now = time.time()
        if key not in self.requests:
            self.requests[key] = []

        self.requests[key] = [t for t in self.requests[key] if now - t < self.window]

        if len(self.requests[key]) >= self.max_requests:
            return False

        self.requests[key].append(now)
        return True

    def remaining(self, key: str) -> int:
        if self._ci_bypass:
            return self.max_requests

        now = time.time()
        if key not in self.requests:
            return self.max_requests
        recent = [t for t in self.requests[key] if now - t < self.window]
        return max(0, self.max_requests - len(recent))


rate_limiter = RateLimiter()
