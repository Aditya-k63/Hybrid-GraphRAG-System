import logging

from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.utils.json_utils import extract_json_object
from app.utils.retry import with_retry

logger = logging.getLogger(__name__)

VERIFY_PROMPT = """Verify whether the candidate answer is grounded in the provided context.

Rules:
- grounded=true if every factual claim in the answer is supported by the context.
- An answer that says it lacks information is grounded=true (honest abstention).
- If the context is empty, the answer cannot be grounded unless it abstains.

Return ONLY JSON: {{"grounded": true|false, "confidence": 0.0-1.0, "note": "brief reason"}}

Context:
{context}

Question: {query}

Answer:
{answer}"""


class Verification(BaseModel):
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    note: str = ""


def _build_prompt(query: str, chunks: list[dict], answer: str) -> str:
    context = "\n\n".join(str(chunk.get("content", "")).strip() for chunk in chunks[:10] if chunk.get("content"))
    context = context[: settings.VERIFY_MAX_CONTEXT_CHARS] or "(no retrieved context)"
    return VERIFY_PROMPT.format(context=context, query=query, answer=answer[:4000])


def _attempt(client, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    raw = extract_json_object(response.choices[0].message.content)
    return Verification(**raw).model_dump()


def verify_answer(
    query: str,
    chunks: list[dict],
    answer: str,
    *,
    client=None,
) -> dict | None:
    """Return {"grounded", "confidence", "note"} or None when verification can't run.

    Never raises: a guard that can't run must not break the answer path.
    """
    if not settings.ENABLE_LIVE_LLM or not settings.ANSWER_VERIFICATION_ENABLED:
        return None

    try:
        from groq import Groq

        client = client or Groq(api_key=settings.GROQ_API_KEY)
        prompt = _build_prompt(query, chunks, answer)
        return with_retry(max_attempts=settings.VERIFY_MAX_RETRIES, base_delay=settings.VERIFY_RETRY_BACKOFF)(
            _attempt
        )(client, prompt)
    except (ValidationError, ValueError) as e:
        logger.warning("Answer verification output invalid: %s", e)
        return None
    except Exception as e:
        logger.warning("Answer verification failed, skipping: %s", e)
        return None