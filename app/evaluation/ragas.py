import logging

logger = logging.getLogger(__name__)


def evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """Evaluate answer quality using the RAGAS library."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        import asyncio
    except ImportError:
        logger.warning("ragas not installed, falling back to simple metrics")
        return _fallback_evaluate(answer, contexts)

    try:
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            ragas_evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
            )
        )
        loop.close()

        row = result[0]
        return {
            "faithfulness": round(float(row.get("faithfulness", 0.0)), 4),
            "answer_relevance": round(float(row.get("answer_relevancy", 0.0)), 4),
            "context_precision": round(float(row.get("context_precision", 0.0)), 4),
            "overall_score": round(
                float(row.get("faithfulness", 0.0)) * 0.4
                + float(row.get("answer_relevancy", 0.0)) * 0.4
                + float(row.get("context_precision", 0.0)) * 0.2,
                4,
            ),
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return _fallback_evaluate(answer, contexts)


def _fallback_evaluate(answer: str, contexts: list[str]) -> dict:
    """Fallback when ragas is unavailable."""
    try:
        from app.ingestion.embedder import get_embedder
        import numpy as np
        model = get_embedder()

        def cos_sim(a, b):
            a, b = np.array(a), np.array(b)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        answer_emb = model.encode(answer)
        context = " ".join(contexts)
        context_emb = model.encode(context)
        faithfulness = round(cos_sim(answer_emb, context_emb), 4)
        q_emb = model.encode(question if "question" in dir() else "")
        a_emb = model.encode(answer)
        relevance = round(cos_sim(q_emb, a_emb), 4)
        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": faithfulness,
            "overall_score": round(faithfulness * 0.4 + relevance * 0.4 + faithfulness * 0.2, 4),
        }
    except Exception:
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0, "overall_score": 0.0}
