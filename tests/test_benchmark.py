"""
RAG Benchmark Regression Tests

Compares current system metrics against a saved baseline to detect regressions.
Used in CI/CD to ensure quality doesn't degrade over time.

Architecture:
- benchmark_results() is a session-scoped fixture that runs the benchmark ONCE
- All assertion tests consume that same result
- This avoids redundant LLM/API calls and makes tests deterministic

Setup: Requires PostgreSQL, Neo4j, and Groq API key.
Run:   python -m pytest tests/test_benchmark.py -v
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).parent.parent / "benchmark_config.json"
BENCHMARK_PATH = Path(__file__).parent.parent / "benchmark_dataset.json"
BASELINE_PATH = Path(__file__).parent.parent / "benchmark_baseline.json"
RESULTS_PATH = Path(__file__).parent.parent / "benchmark_results.json"


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_config() -> dict:
    config = load_json(CONFIG_PATH)
    defaults = {
        "thresholds": {
            "faithfulness": 0.7,
            "answer_relevance": 0.6,
            "context_precision": 0.5,
            "overall_score": 0.6,
        },
        "regression_tolerance": 0.10,
        "weights": {"faithfulness": 0.4, "answer_relevance": 0.4, "context_precision": 0.2},
        "settings": {"top_k": 5, "use_graph": True, "timeout_seconds": 120, "base_url": "http://localhost:8000"},
    }
    for key in defaults:
        if key not in config:
            config[key] = defaults[key]
    return config


def load_baseline() -> dict:
    baseline = load_json(BASELINE_PATH)
    if not baseline or "metrics" not in baseline:
        pytest.skip("No baseline found — run benchmark once to create baseline")
    return baseline


def load_benchmark():
    data = load_json(BENCHMARK_PATH)
    if not data:
        pytest.skip("benchmark_dataset.json not found or empty")
    return data


def get_api_key():
    return os.getenv("API_KEY", "change-me-in-production")


def get_base_url():
    config = load_config()
    return os.getenv("BASE_URL", config["settings"]["base_url"])


def _default_result():
    return {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0,
        "overall_score": 0.0,
        "retrieval_type": "unknown",
    }


def query_system(question: str, base_url: str, api_key: str, max_retries: int = 3) -> dict:
    import requests

    config = load_config()
    timeout = config["settings"].get("timeout_seconds", 120)
    fallback = _default_result()

    for attempt in range(max_retries):
        try:
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
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (500, 429, 503):
                wait = 2 ** (attempt + 1)
                print(f"\n  Retry {attempt + 1}/{max_retries}: got {resp.status_code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            wait = 2 ** (attempt + 1)
            print(f"\n  Retry {attempt + 1}/{max_retries}: connection failed, waiting {wait}s...")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"\n  WARNING: {e}")
            return fallback

    print(f"\n  FAILED after {max_retries} retries for: {question[:50]}...")
    return fallback


def save_results(results: dict):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def assert_no_regression(metric_name: str, current: float, baseline: float, tolerance: float):
    drop = baseline - current
    pct_drop = (drop / baseline * 100) if baseline > 0 else 0
    assert current >= baseline - tolerance, (
        f"\n  REGRESSION DETECTED: {metric_name}\n"
        f"  Baseline:  {baseline:.4f}\n"
        f"  Current:   {current:.4f}\n"
        f"  Drop:      {drop:.4f} ({pct_drop:.1f}%)\n"
        f"  Tolerance: {tolerance:.4f}"
    )


def _run_benchmark() -> dict:
    """Run the benchmark once and return results."""
    config = load_config()
    benchmark = load_benchmark()
    base_url = get_base_url()
    api_key = get_api_key()

    scores = {"faithfulness": [], "answer_relevance": [], "context_precision": [], "overall_score": []}
    per_question = []
    errors = []

    for i, item in enumerate(benchmark):
        if i > 0:
            time.sleep(2)
        result = query_system(item["question"], base_url, api_key)
        for key in scores:
            scores[key].append(result.get(key, 0.0))
        if result.get("error"):
            errors.append({"question": item["question"], "error": result["error"]})
        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "expected_retrieval": item.get("expected_retrieval"),
            "actual_retrieval": result.get("retrieval_type"),
            "faithfulness": result.get("faithfulness", 0.0),
            "answer_relevance": result.get("answer_relevance", 0.0),
            "context_precision": result.get("context_precision", 0.0),
            "overall_score": result.get("overall_score", 0.0),
        })

    avg = {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in scores.items()}

    if errors:
        print(f"\n  ERRORS ({len(errors)}/{len(benchmark)}):")
        for e in errors:
            print(f"    - {e['question'][:50]}... -> {e['error']}")

    return {"avg": avg, "per_question": per_question, "raw": scores, "errors": errors}


# ──────────────────────────────────────────────
#  Session-scoped fixture: run benchmark ONCE
# ──────────────────────────────────────────────
@pytest.fixture(scope="session")
def benchmark_results():
    results = _run_benchmark()
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": results["avg"],
        "per_question": results["per_question"],
        "errors": results.get("errors", []),
    }
    save_results(output)
    print(f"\n  Benchmark results saved to {RESULTS_PATH}")
    return results


# ──────────────────────────────────────────────
#  Health check
# ──────────────────────────────────────────────
def test_health():
    import requests

    base_url = get_base_url()
    resp = requests.get(f"{base_url}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded"), f"System unhealthy: {data}"


# ──────────────────────────────────────────────
#  Regression tests (all consume same fixture)
# ──────────────────────────────────────────────
def test_no_regression_faithfulness(benchmark_results):
    baseline = load_baseline()
    config = load_config()
    tolerance = config.get("regression_tolerance", 0.10)
    current = benchmark_results["avg"]["faithfulness"]
    base = baseline["metrics"]["faithfulness"]
    print(f"\n  faithfulness: current={current:.4f}, baseline={base:.4f}")
    assert_no_regression("faithfulness", current, base, tolerance)


def test_no_regression_answer_relevance(benchmark_results):
    baseline = load_baseline()
    config = load_config()
    tolerance = config.get("regression_tolerance", 0.10)
    current = benchmark_results["avg"]["answer_relevance"]
    base = baseline["metrics"]["answer_relevance"]
    print(f"\n  answer_relevance: current={current:.4f}, baseline={base:.4f}")
    assert_no_regression("answer_relevance", current, base, tolerance)


def test_no_regression_context_precision(benchmark_results):
    baseline = load_baseline()
    config = load_config()
    tolerance = config.get("regression_tolerance", 0.10)
    current = benchmark_results["avg"]["context_precision"]
    base = baseline["metrics"]["context_precision"]
    print(f"\n  context_precision: current={current:.4f}, baseline={base:.4f}")
    assert_no_regression("context_precision", current, base, tolerance)


def test_no_regression_overall(benchmark_results):
    baseline = load_baseline()
    config = load_config()
    tolerance = config.get("regression_tolerance", 0.10)
    current = benchmark_results["avg"]["overall_score"]
    base = baseline["metrics"]["overall_score"]
    print(f"\n  overall_score: current={current:.4f}, baseline={base:.4f}")
    assert_no_regression("overall_score", current, base, tolerance)


def test_save_results(benchmark_results):
    assert RESULTS_PATH.exists(), "benchmark_results.json should have been created by fixture"
    data = load_json(RESULTS_PATH)
    assert "metrics" in data
    assert "per_question" in data
    print(f"\n  Results verified at {RESULTS_PATH}")
