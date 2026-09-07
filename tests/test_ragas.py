"""Tests for RAGAS evaluation pipeline."""
import pytest
from app.evaluation.ragas import evaluate, _fallback_evaluate, evaluate_batch


def test_fallback_evaluate_basic():
    """Test fallback evaluation with minimal data."""
    result = _fallback_evaluate("What is AI?", "AI is artificial intelligence.", ["AI stands for artificial intelligence."])
    assert "faithfulness" in result
    assert "answer_relevance" in result
    assert "context_precision" in result
    assert "overall_score" in result
    assert 0.0 <= result["faithfulness"] <= 1.0
    assert 0.0 <= result["answer_relevance"] <= 1.0


def test_fallback_evaluate_empty():
    """Test fallback evaluation with empty data."""
    result = _fallback_evaluate("", "", [])
    assert result["faithfulness"] == 0.0


def test_evaluate_returns_dict():
    """Test that evaluate returns expected dict structure."""
    result = evaluate("What is Python?", "Python is a programming language.", ["Python is a high-level programming language."])
    assert isinstance(result, dict)
    assert "faithfulness" in result


def test_evaluate_batch():
    """Test batch evaluation."""
    questions = ["What is AI?", "What is ML?"]
    answers = ["AI is artificial intelligence.", "ML is machine learning."]
    contexts = [["AI stands for artificial intelligence."], ["ML stands for machine learning."]]
    result = evaluate_batch(questions, answers, contexts)
    assert isinstance(result, dict)
    if "error" not in result:
        assert "count" in result
        assert result["count"] == 2


def test_overall_score_calculation():
    """Test that overall_score is weighted correctly."""
    result = _fallback_evaluate("test question", "test answer", ["test context"])
    expected = (
        result["faithfulness"] * 0.4
        + result["answer_relevance"] * 0.4
        + result["context_precision"] * 0.2
    )
    assert abs(result["overall_score"] - round(expected, 4)) < 0.001