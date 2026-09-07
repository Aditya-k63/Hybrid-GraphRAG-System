import logging
from typing import Any

logger = logging.getLogger(__name__)

RAGAS_AVAILABLE = False
_datasets_available = False

try:
    import ragas
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("ragas not installed. Install with: pip install ragas")

try:
    from datasets import Dataset
    _datasets_available = True
except ImportError:
    logger.warning("datasets not installed. Install with: pip install datasets")


def _get_llm_for_ragas():
    """Get a LangChain-compatible LLM for RAGAS evaluation using Groq."""
    try:
        from langchain_groq import ChatGroq
        from app.config import settings
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.0,
            max_tokens=1024,
        )
    except ImportError:
        logger.error("langchain-groq not installed for RAGAS")
        return None
    except Exception as e:
        logger.error(f"Failed to create RAGAS LLM: {e}")
        return None


def _get_embeddings_for_ragas():
    """Get embeddings for RAGAS evaluation."""
    try:
        from ragas.embeddings import HuggingfaceEmbeddings
        from app.config import settings
        return HuggingfaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    except ImportError:
        try:
            from ragas.embeddings import LangchainEmbeddings
            from app.ingestion.embedder import get_embedder
            model = get_embedder()
            return LangchainEmbeddings(model=model)
        except Exception:
            logger.warning("Using default RAGAS embeddings")
            return None


def evaluate(question: str, answer: str, contexts: list[str], ground_truth: str = "") -> dict:
    """Evaluate answer quality using the RAGAS library with Groq as the LLM.

    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of retrieved context chunks
        ground_truth: Optional reference answer for context_recall
    """
    if not RAGAS_AVAILABLE:
        logger.warning("ragas not installed, using fallback")
        return _fallback_evaluate(question, answer, contexts)

    if not _datasets_available:
        logger.warning("datasets not installed, using fallback")
        return _fallback_evaluate(question, answer, contexts)

    llm = _get_llm_for_ragas()
    if llm is None:
        logger.warning("LLM not available for RAGAS, using fallback")
        return _fallback_evaluate(question, answer, contexts)

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        metrics_to_use = [faithfulness, answer_relevancy, context_precision]

        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=llm,
        )

        row = result[0]
        scores = {}
        for key in ["faithfulness", "answer_relevancy", "context_precision"]:
            val = row.get(key, 0.0)
            scores[key] = round(float(val) if val is not None else 0.0, 4)

        return {
            "faithfulness": scores["faithfulness"],
            "answer_relevance": scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "overall_score": round(
                scores["faithfulness"] * 0.4
                + scores["answer_relevancy"] * 0.4
                + scores["context_precision"] * 0.2,
                4,
            ),
        }

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return _fallback_evaluate(question, answer, contexts)


def evaluate_batch(questions: list[str], answers: list[str], contexts_list: list[list[str]]) -> dict:
    """Evaluate a batch of Q&A pairs using RAGAS.

    Returns aggregate metrics across all questions.
    """
    if not RAGAS_AVAILABLE or not _datasets_available:
        return {"error": "ragas or datasets not installed"}

    llm = _get_llm_for_ragas()
    if llm is None:
        return {"error": "LLM not available"}

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        })

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
        )

        scores = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
        }
        for row in result:
            for key in scores:
                val = row.get(key, 0.0)
                scores[key].append(float(val) if val is not None else 0.0)

        return {
            "count": len(questions),
            "faithfulness": round(sum(scores["faithfulness"]) / len(scores["faithfulness"]), 4),
            "answer_relevance": round(sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"]), 4),
            "context_precision": round(sum(scores["context_precision"]) / len(scores["context_precision"]), 4),
            "overall_score": round(
                sum(scores["faithfulness"]) / len(scores["faithfulness"]) * 0.4
                + sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"]) * 0.4
                + sum(scores["context_precision"]) / len(scores["context_precision"]) * 0.2,
                4,
            ),
            "per_question": [
                {
                    "question": q,
                    "faithfulness": round(scores["faithfulness"][i], 4),
                    "answer_relevance": round(scores["answer_relevancy"][i], 4),
                    "context_precision": round(scores["context_precision"][i], 4),
                }
                for i, q in enumerate(questions)
            ],
        }
    except Exception as e:
        logger.error(f"Batch RAGAS evaluation failed: {e}")
        return {"error": str(e)}


def _fallback_evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """Fallback when ragas is unavailable — uses embedding similarity."""
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

        q_emb = model.encode(question)
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