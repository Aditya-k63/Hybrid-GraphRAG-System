import logging
import json

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


def classify_query(query: str) -> dict:
    try:
        from app.config import settings
        from app.retrieval.tracker import tracker
        client = get_groq()
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": CLASSIFY_PROMPT.format(query=query),
            }],
            temperature=0.1,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
        result = json.loads(content)
        retrieval_type = result.get("type", "hybrid")
        if retrieval_type not in ("vector", "graph", "hybrid", "direct"):
            retrieval_type = "hybrid"
        tracker.record("classifier", 1)
        return {"type": retrieval_type, "reason": result.get("reason", ""), "retrieval_type": retrieval_type}
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        return {"type": "hybrid", "reason": "classification failed, defaulting to hybrid", "retrieval_type": "hybrid"}


def classify_and_count(query: str) -> dict:
    """Classify query and track retrieval call reduction."""
    result = classify_query(query)
    from app.retrieval.tracker import tracker
    tracker.record(result["type"], 1)
    return result