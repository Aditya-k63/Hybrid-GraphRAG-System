"""
RAG Benchmark Regression Tests

Runs the benchmark dataset against the system and checks that quality
metrics stay above minimum thresholds. Used in CI/CD to catch regressions.

Setup: Requires PostgreSQL, Neo4j, and Groq API key.
Run:   python -m pytest tests/test_benchmark.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).parent.parent / "benchmark_config.json"
BENCHMARK_PATH = Path(__file__).parent.parent / "benchmark_dataset.json"


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {
        "thresholds": {
            "faithfulness": 0.7,
            "answer_relevance": 0.6,
            "context_precision": 0.5,
            "overall_score": 0.6,
            "classifier_accuracy": 0.5,
        },
        "weights": {"faithfulness": 0.4, "answer_relevance": 0.4, "context_precision": 0.2},
        "settings": {"top_k": 5, "use_graph": True, "timeout_seconds": 120, "base_url": "http://localhost:8000"},
    }


def load_benchmark():
    if not BENCHMARK_PATH.exists():
        pytest.skip("benchmark_dataset.json not found")
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def get_api_key():
    return os.getenv("API_KEY", "change-me-in-production")


def get_base_url():
    config = load_config()
    return os.getenv("BASE_URL", config["settings"]["base_url"])


def query_system(question: str, base_url: str, api_key: str) -> dict:
    import requests

    config = load_config()
    timeout = config["settings"].get("timeout_seconds", 120)
    resp = requests.post(
        f"{base_url}/evaluate-query",
        json={
            "question": question,
            "top_k": config["settings"]["top_k"],
            "use_graph": config["settings"]["use_graph"],
        },
        headers={"X-API-Key": api_key},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def test_health():
    import requests

    base_url = get_base_url()
    resp = requests.get(f"{base_url}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded"), f"System unhealthy: {data}"


def test_benchmark_faithfulness():
    config = load_config()
    threshold = config["thresholds"]["faithfulness"]
    benchmark = load_benchmark()
    base_url = get_base_url()
    api_key = get_api_key()

    scores = []
    for item in benchmark:
        result = query_system(item["question"], base_url, api_key)
        scores.append(result.get("faithfulness", 0.0))

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nFaithfulness scores: {[round(s, 3) for s in scores]}")
    print(f"Average faithfulness: {avg:.3f} (threshold: {threshold})")
    assert avg >= threshold, f"Faithfulness {avg:.3f} below threshold {threshold}"


def test_benchmark_answer_relevance():
    config = load_config()
    threshold = config["thresholds"]["answer_relevance"]
    benchmark = load_benchmark()
    base_url = get_base_url()
    api_key = get_api_key()

    scores = []
    for item in benchmark:
        result = query_system(item["question"], base_url, api_key)
        scores.append(result.get("answer_relevance", 0.0))

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nAnswer relevance scores: {[round(s, 3) for s in scores]}")
    print(f"Average answer relevance: {avg:.3f} (threshold: {threshold})")
    assert avg >= threshold, f"Answer relevance {avg:.3f} below threshold {threshold}"


def test_benchmark_context_precision():
    config = load_config()
    threshold = config["thresholds"]["context_precision"]
    benchmark = load_benchmark()
    base_url = get_base_url()
    api_key = get_api_key()

    scores = []
    for item in benchmark:
        result = query_system(item["question"], base_url, api_key)
        scores.append(result.get("context_precision", 0.0))

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nContext precision scores: {[round(s, 3) for s in scores]}")
    print(f"Average context precision: {avg:.3f} (threshold: {threshold})")
    assert avg >= threshold, f"Context precision {avg:.3f} below threshold {threshold}"


def test_benchmark_overall():
    config = load_config()
    threshold = config["thresholds"]["overall_score"]
    benchmark = load_benchmark()
    base_url = get_base_url()
    api_key = get_api_key()

    scores = []
    failures = []
    for item in benchmark:
        result = query_system(item["question"], base_url, api_key)
        overall = result.get("overall_score", 0.0)
        scores.append(overall)
        if overall < threshold:
            failures.append({
                "question": item["question"],
                "score": overall,
                "expected_retrieval": item.get("expected_retrieval"),
                "actual_retrieval": result.get("retrieval_type"),
            })

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nOverall scores: {[round(s, 3) for s in scores]}")
    print(f"Average overall: {avg:.3f} (threshold: {threshold})")
    if failures:
        print(f"Failed questions ({len(failures)}):")
        for f in failures:
            print(f"  - {f['question'][:60]}... (score: {f['score']:.3f})")

    assert avg >= threshold, f"Overall score {avg:.3f} below threshold {threshold}"


def test_retrieval_classifier_accuracy():
    config = load_config()
    threshold = config["thresholds"]["classifier_accuracy"]
    benchmark = load_benchmark()
    base_url = get_base_url()
    api_key = get_api_key()

    correct = 0
    total = 0
    results = []
    for item in benchmark:
        expected = item.get("expected_retrieval")
        if not expected:
            continue
        result = query_system(item["question"], base_url, api_key)
        actual = result.get("retrieval_type", "unknown")
        match = actual == expected
        if match:
            correct += 1
        total += 1
        results.append({
            "question": item["question"][:50],
            "expected": expected,
            "actual": actual,
            "match": match,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nClassifier accuracy: {accuracy:.1%} ({correct}/{total}) (threshold: {threshold:.0%})")
    for r in results:
        status = "OK" if r["match"] else "MISS"
        print(f"  [{status}] {r['question']}... expected={r['expected']}, got={r['actual']}")

    assert accuracy >= threshold, f"Classifier accuracy {accuracy:.1%} below {threshold:.0%} threshold"
