"""Regression gate: compare a candidate benchmark run against the baseline.

Usage (after producing a candidate run file, e.g. benchmark_results_crag.json):
    python scripts/compare_baseline.py \
        --baseline benchmark_baseline.json \
        --candidate benchmark_results_crag.json

Both files follow the benchmark baseline schema (metrics + per_question).
Exit code 0 = gate passed (merge may proceed);
exit code 1 = regression detected on at least one metric (block the merge).
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_TOLERANCE = 0.10  # mirrors benchmark_config.json regression_tolerance
METRICS = ("faithfulness", "answer_relevance", "context_precision",
           "overall_score")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tolerance(baseline: dict) -> float:
    """Prefer the repo's benchmark_config.json value; fall back to default."""
    cfg_path = Path(__file__).resolve().parent.parent / "benchmark_config.json"
    try:
        cfg = load_json(cfg_path)
        return float(cfg.get("regression_tolerance", DEFAULT_TOLERANCE))
    except (OSError, ValueError, TypeError):
        return DEFAULT_TOLERANCE


def compare(
    baseline: dict,
    candidate: dict,
    tolerance: float,
) -> tuple[bool, list[str]]:
    """Compare aggregate metrics. Returns (passed, report lines)."""
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    lines: list[str] = []
    passed = True

    for name in METRICS:
        before = float(base_metrics.get(name, 0.0))
        after = float(cand_metrics.get(name, 0.0))
        delta = round(after - before, 4)
        verdict = "PASS" if delta >= -tolerance else "FAIL"
        if delta < -tolerance:
            passed = False
        lines.append(
            f"{name:>18}: {before:.4f} -> {after:.4f} "
            f"(delta {delta:+.4f}) [{verdict}, floor -{tolerance:.2f}]"
        )
    return passed, lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark regression gate (baseline vs candidate)")
    parser.add_argument("--baseline", default="benchmark_baseline.json")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Override regression tolerance (e.g. 0.02)")
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline))
    candidate = load_json(Path(args.candidate))
    tolerance = (args.tolerance if args.tolerance is not None
                 else read_tolerance(baseline))

    passed, lines = compare(baseline, candidate, tolerance)
    print("=" * 60)
    print("REGRESSION GATE REPORT")
    print("=" * 60)
    for line in lines:
        print(f"  {line}")
    print("=" * 60)
    if passed:
        print(f"Gate PASSED: no metric regressed beyond -{tolerance:.2f}.")
        print("Candidate version may be considered for production promotion.")
        return 0
    print("Gate FAILED: candidate regresses on at least one metric.")
    print("Keep the current version; do not promote the candidate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
