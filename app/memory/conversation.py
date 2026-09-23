import json
import logging
import re
import time
from collections import defaultdict

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

SELECT_MEMORY_PROMPT = """You are selecting turns from a conversation to use as context for answering a NEW question.

Rules:
- Keep only turns that help answer the new question: turns mentioning the same entities, topics, constraints, or previously chosen options.
- Drop unrelated small talk, status updates, and irrelevant tangents.
- Keep earlier turns only when they establish a fact the question depends on.
- Return ONLY valid JSON: {{"keep": [0, 2, 5]}} listing the indices of the turns to keep.

Turn log:
{turns}

New question: {query}"""

_STOPWORD_FILTER = {
    "what", "which", "how", "does", "would", "could", "should", "can", "the", "a", "an",
    "is", "are", "was", "were", "do", "did", "i", "you", "we", "they", "it", "to", "of",
    "in", "for", "on", "with", "and", "or", "not", "this", "that", "from", "as", "by",
    "at", "about", "have", "has", "had", "be", "been", "if", "then", "so", "me", "my",
}


def _parse_keep_indices(content: str) -> list[int] | None:
    """Defensively parse the selector response; returns None for garbage."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return None
    keep = parsed.get("keep") if isinstance(parsed, dict) else None
    if not isinstance(keep, list):
        return None
    indices = [int(item) for item in keep if isinstance(item, (int, float, str)) and str(item).lstrip("-").isdigit()]
    return indices or None


def _lexical_score(query: str, content: str) -> int:
    tokens = [t for t in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(t) > 2 and t not in _STOPWORD_FILTER]
    if not tokens:
        return 0
    haystack = content.lower()
    return sum(haystack.count(token) for token in tokens)


class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.sessions: dict[str, list[dict]] = defaultdict(list)
        self.last_seen: dict[str, float] = defaultdict(float)

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content})
        self.last_seen[session_id] = time.monotonic()
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)
        self.last_seen.pop(session_id, None)

    def _is_expired(self, session_id: str, now: float) -> bool:
        last = self.last_seen.get(session_id)
        return last is not None and (now - last) > settings.MEMORY_TTL_SECONDS

    def get_summary(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""
        turns = []
        for msg in history[-6:]:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            turns.append(f"{prefix}: {msg['content'][:200]}")
        return "\n".join(turns)

    def select_relevant(
        self,
        session_id: str,
        query: str,
        top_k: int | None = None,
        max_tokens: int | None = None,
        now: float | None = None,
    ) -> list[dict]:
        """Return only the history turns relevant to the current question.

        Uses an LLM selector (find-relevant-memories pattern) with a deterministic
        lexical fallback so CI and offline runs never depend on the model provider.
        Sessions idle past MEMORY_TTL_SECONDS are dropped on access.
        """
        now = now or time.monotonic()
        if self._is_expired(session_id, now):
            self.clear(session_id)
            return []

        history = self.get_history(session_id)
        if not history:
            return []
        self.last_seen[session_id] = now

        top_k = top_k or settings.MEMORY_SELECT_TOP_K
        max_tokens = max_tokens or settings.MEMORY_MAX_CONTEXT_TOKENS

        selected = self._select_via_llm(history, query, top_k)
        if selected is None:
            selected = self._select_fallback(history, query, top_k)
        return self._cap_tokens(selected, max_tokens)

    def _select_via_llm(self, history: list[dict], query: str, top_k: int) -> list[dict] | None:
        if not settings.ENABLE_LIVE_LLM:
            return None
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)
            turns = "\n".join(f"[{i}] {msg['role']}: {msg['content'][:300]}" for i, msg in enumerate(history))
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=[{
                            "role": "user",
                            "content": SELECT_MEMORY_PROMPT.format(turns=turns, query=query),
                        }],
                        temperature=0.1,
                        max_tokens=200,
                        response_format={"type": "json_object"},
                    )
                    indices = _parse_keep_indices(response.choices[0].message.content)
                    if indices is None:
                        raise ValueError("selector returned no valid indices")
                    valid = sorted({i for i in indices if 0 <= i < len(history)})[:top_k]
                    return [history[i] for i in valid] if valid else None
                except Exception as e:
                    last_error = e
                    if attempt < 1:
                        time.sleep(2 ** attempt)
                        continue
            logger.warning("Memory selector failed, falling back to lexical selection: %s", last_error)
            return None
        except Exception as e:
            logger.warning("Memory selector unavailable, falling back to lexical selection: %s", e)
            return None

    def _select_fallback(self, history: list[dict], query: str, top_k: int) -> list[dict]:
        scored = [(i, _lexical_score(query, msg["content"])) for i, msg in enumerate(history)]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        usable = [i for i, score in ranked if score > 0]
        if not usable:
            usable = list(range(max(0, len(history) - top_k), len(history)))
        picked = sorted(set(usable[:top_k]))
        return [history[i] for i in picked]

    def _cap_tokens(self, messages: list[dict], max_tokens: int) -> list[dict]:
        if not messages:
            return []
        estimate = sum(len(msg["content"]) for msg in messages) // 4
        if estimate <= max_tokens:
            return messages
        kept = messages[:]
        while kept and estimate > max_tokens and len(kept) > 1:
            removed = kept.pop(0)
            estimate -= len(removed["content"]) // 4
        if kept and estimate > max_tokens:
            allowed_chars = max_tokens * 4
            kept[-1] = {**kept[-1], "content": kept[-1]["content"][-allowed_chars:]}
        return kept


memory = ConversationMemory()