import logging
from typing import Any

logger = logging.getLogger(__name__)

RAGAS_AVAILABLE = False
_datasets_available = False

try:
    import ragas
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ragas metrics unavailable: {e}. Using semantic similarity fallback")

try:
    from datasets import Dataset
    _datasets_available = True
except ImportError:
    logger.warning("datasets not installed. Install with: pip install datasets")


def _get_llm_for_ragas():
    try:
        from langchain_groq import ChatGroq
        from app.config import settings
        return ChatGroq(groq_api_key=settings.GROQ_API_KEY, model=settings.LLM_MODEL, temperature=0.0, max_tokens=1024)
    except Exception as e:
        logger.error(f"Failed to create RAGAS LLM: {e}")
        return None


def _get_embeddings_for_ragas():
    try:
        from ragas.embeddings import HuggingfaceEmbeddings
        from app.config import settings
        return HuggingfaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    except ImportError:
        try:
            from ragas.embeddings import LangchainEmbeddings
            from app.ingestion.embedder import get_embedder
            return LangchainEmbeddings(model=get_embedder())
        except Exception:
            return None


def _fallback_evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    return semantic_similarity_evaluate(question, answer, contexts)


def _extract_result_row(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    try:
        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
            if not frame.empty:
                return frame.iloc[0].to_dict()
    except Exception:
        pass
    try:
        return result[0]
    except Exception:
        return {}


def evaluate(question: str, answer: str, contexts: list[str], ground_truth: str = "") -> dict:
    """Evaluate with RAGAS in production and a deterministic embedding proxy in CI."""
    from app.config import settings

    if not settings.ENABLE_LIVE_LLM:
        return _fallback_evaluate(question, answer, contexts)
    if not RAGAS_AVAILABLE or not _datasets_available:
        return _fallback_evaluate(question, answer, contexts)

    llm = _get_llm_for_ragas()
    if llm is None:
        return _fallback_evaluate(question, answer, contexts)

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        data = {"question": [question], "answer": [answer], "contexts": [contexts]}
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        result = ragas_evaluate(
            dataset=Dataset.from_dict(data),
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
        )
        row = _extract_result_row(result)
        scores = {}
        for key in ["faithfulness", "answer_relevancy", "context_precision"]:
            val = row.get(key, 0.0)
            scores[key] = round(float(val) if val is not None else 0.0, 4)
        return {
            "faithfulness": scores["faithfulness"],
            "answer_relevance": scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "overall_score": round(scores["faithfulness"] * 0.4 + scores["answer_relevancy"] * 0.4 + scores["context_precision"] * 0.2, 4),
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return _fallback_evaluate(question, answer, contexts)


def evaluate_batch(questions: list[str], answers: list[str], contexts_list: list[list[str]]) -> dict:
    if not RAGAS_AVAILABLE or not _datasets_available:
        return {"error": "ragas or datasets not installed"}
    llm = _get_llm_for_ragas()
    if llm is None:
        return {"error": "LLM not available"}
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        result = ragas_evaluate(
            dataset=Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts_list}),
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
        )
        rows = result.to_pandas().to_dict(orient="records") if hasattr(result, "to_pandas") else list(result)
        scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}
        for row in rows:
            for key in scores:
                val = row.get(key, 0.0)
                scores[key].append(float(val) if val is not None else 0.0)
        if not scores["faithfulness"]:
            return {"error": "RAGAS returned no evaluation rows"}
        f = sum(scores["faithfulness"]) / len(scores["faithfulness"])
        a = sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"])
        c = sum(scores["context_precision"]) / len(scores["context_precision"])
        return {
            "count": len(questions), "faithfulness": round(f, 4), "answer_relevance": round(a, 4),
            "context_precision": round(c, 4), "overall_score": round(f * 0.4 + a * 0.4 + c * 0.2, 4),
            "per_question": [{"question": q, "faithfulness": round(scores["faithfulness"][i], 4), "answer_relevance": round(scores["answer_relevancy"][i], 4), "context_precision": round(scores["context_precision"][i], 4)} for i, q in enumerate(questions)],
        }
    except Exception as e:
        logger.error(f"Batch RAGAS evaluation failed: {e}")
        return {"error": str(e)}


def semantic_similarity_evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    if not question or not answer or not contexts:
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0, "overall_score": 0.0}
    try:
        from app.ingestion.embedder import get_embedder
        import numpy as np
        model = get_embedder()
        def cos_sim(a, b):
            a, b = np.array(a), np.array(b)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
        answer_context_sim = round(cos_sim(model.encode(answer), model.encode(" ".join(contexts))), 4)
        question_answer_sim = round(cos_sim(model.encode(question), model.encode(answer)), 4)
        return {
            "faithfulness": answer_context_sim,
            "answer_relevance": question_answer_sim,
            "context_precision": answer_context_sim,
            "overall_score": round(answer_context_sim * 0.4 + question_answer_sim * 0.4 + answer_context_sim * 0.2, 4),
        }
    except Exception:
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0, "overall_score": 0.0}
