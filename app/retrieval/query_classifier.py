import json
import logging
import re
import time

logger = logging.getLogger(__name__)


def get_groq():
    from groq import Groq
    from app.config import settings
    return Groq(api_key=settings.GROQ_API_KEY)


CLASSIFY_PROMPT = """Classify the following question into one of these retrieval types:
- "vector": General questions that need semantic search (definitions, explanations, summaries)
- "graph": Questions about specific entities, relationships, or connections between things
- "hybrid": Complex questions that need both factual/semantic info AND entity relationships
- "direct": Simple factual questions that can be answered directly

Return ONLY a JSON object: {{"type": "...", "reason": "..."}}

Question: {query}"""


def _parse_json_response(content: str) -> dict:
    """Parse a classifier response defensively and always return a valid classification."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("classifier response is not a JSON object")
    except Exception:
        return {"type": "hybrid", "reason": "invalid model output", "retrieval_type": "hybrid"}

    retrieval_type = result.get("type", "hybrid")
    if retrieval_type not in ("vector", "graph", "hybrid", "direct"):
        retrieval_type = "hybrid"
    return {
        "type": retrieval_type,
        "reason": result.get("reason", ""),
        "retrieval_type": retrieval_type,
    }


def classify_query(query: str) -> dict:
    try:
        from app.config import settings
        from app.retrieval.tracker import tracker

        if not settings.ENABLE_LIVE_LLM:
            return {"type": "hybrid", "reason": "LLM disabled in CI", "retrieval_type": "hybrid"}

        client = get_groq()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": CLASSIFY_PROMPT.format(query=query),
                    }],
                    temperature=0.1,
                    max_tokens=100,
                )
                content = (response.choices[0].message.content or "").strip()
                result = _parse_json_response(content)
                tracker.record("classifier", 1)
                return result
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status == 429 or "429" in str(e):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                logger.error(f"Query classification failed: {e}")
                return {"type": "hybrid", "reason": "classification failed, defaulting to hybrid", "retrieval_type": "hybrid"}

        return {"type": "hybrid", "reason": "classification retries exhausted", "retrieval_type": "hybrid"}
    except Exception as e:
        logger.error(f"Query classification setup failed: {e}")
        return {"type": "hybrid", "reason": "classification failed, defaulting to hybrid", "retrieval_type": "hybrid"}


def classify_and_count(query: str) -> dict:
    """Classify query and track retrieval call reduction."""
    result = classify_query(query)
    from app.retrieval.tracker import tracker
    tracker.record(result["type"], 1)
    return result
