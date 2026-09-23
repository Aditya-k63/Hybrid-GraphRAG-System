CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Rough token estimate (Claude toolSearch.ts style): characters / chars-per-token."""
    return max(0, int(len(text or "") / chars_per_token))


def chars_for_tokens(token_count: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    return max(1, int(token_count * chars_per_token))