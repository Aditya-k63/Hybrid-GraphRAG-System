import logging

from app.utils.json_utils import extract_json_object
from app.utils.retry import with_retry

logger = logging.getLogger(__name__)

RETRIEVAL_TYPES = ("vector", "graph", "hybrid", "direct")


def get_groq():
    from groq import Groq
    from app.config import settings

    return Groq(api_key=settings.GROQ_API_KEY)


CLASSIFY_PROMPT = """Classify the following question into one of these retrieval types:
- "vector": General questions that need semantic search (definitions, explanations, summaries)
- "graph": Questions about specific entities, relationships, or connections between things
- "hybrid": Complex questions that need both factual/semantic info AND entity relationships
- "direct": Simple factual questions that can be answered directly

Return ONLY a JSON object: {"type": "...", "reason": "..."}

Question: {query}"""


def _parse_json_response(content: str) -> dict:
    """Parse a classifier response defensively and always return a valid classification."""
    try:
        result = extract_json_object(content)
    except Exception:
        return {"type": "hybrid", "reason": "invalid model output", "retrieval_type": "hybrid"}

    retrieval_type = result.get("type", "hybrid")
    if retrieval_type not in RETRIEVAL_TYPES:
        retrieval_type = "hybrid"
    return {
        "type": retrieval_type,
        "reason": str(result.get("reason", "")),
        "retrieval_type": retrieval_type,
    }


def classify_query(query: str) -> dict:
    try:
        from app.config import settings
        from app.retrieval.tracker import tracker

        if not settings.ENABLE_LIVE_LLM:
            return {"type": "hybrid", "reason": "LLM disabled in CI", "retrieval_type": "hybrid"}

        client = get_groq()

        @with_retry(max_attempts=settings.CLASSIFIER_MAX_RETRIES, base_delay=2.0)
        def attempt() -> dict:
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            return _parse_json_response(response.choices[0].message.content)

        result = attempt()
        tracker.record("classifier", 1)
        return result
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        return {"type": "hybrid", "reason": "classification failed, defaulting to hybrid", "retrieval_type": "hybrid"}


def classify_and_count(query: str) -> dict:
    """Classify query and track retrieval call reduction."""
    result = classify_query(query)
    from app.retrieval.tracker import tracker

    tracker.record(result["type"], 1)
    return result