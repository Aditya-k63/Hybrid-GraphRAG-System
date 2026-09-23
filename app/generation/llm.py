import logging

logger = logging.getLogger(__name__)


def get_groq():
    from groq import Groq
    from app.config import settings
    return Groq(api_key=settings.GROQ_API_KEY)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Rules:
- Answer ONLY using the information in the context below.
- Be specific and detailed. Include entity names, dates, and relationships when available.
- If the context is partially relevant, share what you know and flag what's missing.
- If the answer is completely absent, say: "I don't have enough information to answer that."
- Never make up facts not present in the context.
- When citing sources, reference the document name if available."""


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    """Deterministic answer used by CI and when the external LLM is unavailable."""
    if not chunks:
        return "I don't have enough information to answer that."
    return "Based on the retrieved context:\n\n" + "\n\n".join(
        f"[{i}] {chunk.get('content', '').strip()}"
        for i, chunk in enumerate(chunks, 1)
        if chunk.get("content")
    )


def generate_answer(query: str, chunks: list[dict], conversation_history: list[dict] = None) -> str:
    from app.config import settings

    if not settings.ENABLE_LIVE_LLM:
        return _fallback_answer(query, chunks)

    client = get_groq()

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = ""
        if isinstance(chunk.get("meta"), dict):
            source = f" [{chunk['meta'].get('source', '')}]"
        context_parts.append(f"[{i}]{source}\n{chunk['content']}")

    context = "\n\n".join(context_parts)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
    })

    from app.utils.retry import with_retry

    @with_retry(max_attempts=3, base_delay=2.0)
    def _completion() -> str:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("LLM returned an empty response")
        return content.strip()

    try:
        return _completion()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return _fallback_answer(query, chunks)
