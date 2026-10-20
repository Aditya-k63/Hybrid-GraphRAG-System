"""Compare hybrid vs crag on the golden + unanswerable question sets (live).

Usage (server running with ENABLE_CRAG=true and a live LLM):
    python scripts/run_crag_comparison.py --base-url http://127.0.0.1:8000

Runs /evaluate-query twice per question (mode=hybrid and mode=crag) so both
modes use the SAME RAGAS-based evaluation implementation.

Output artifact (benchmark_results_crag.json) is schema-compatible with
benchmark_baseline.json: "metrics" holds the CRAG (candidate) means over all
questions, so scripts/compare_baseline.py can gate it directly:
    python scripts/compare_baseline.py --candidate benchmark_results_crag.json
"""
import argparse
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

UNANSWERABLE = [
    {"id": 100, "question": "What is the capital city of France?",
     "question_type": "unanswerable"},
    {"id": 101, "question": "Who wrote the novel Pride and Prejudice?",
     "question_type": "unanswerable"},
    {"id": 102, "question": "How tall is the Eiffel Tower in meters?",
     "question_type": "unanswerable"},
    {"id": 103, "question": "What is the recipe for pasta carbonara?",
     "question_type": "unanswerable"},
    {"id": 104, "question": "Who won the FIFA World Cup in 2010?",
     "question_type": "unanswerable"},
]

METRIC_KEYS = ("faithfulness", "answer_relevance", "context_precision",
               "overall_score")


def load_questions() -> dict:
    dataset = json.loads(
        (Path(__file__).parent.parent / "benchmark_dataset.json")
        .read_text(encoding="utf-8"))
    golden = [{"id": q["id"], "question": q["question"],
               "question_type": str(q.get("question_type", "golden"))}
              for q in dataset]
    return {"answerable": golden, "unanswerable": UNANSWERABLE}


def run_mode(session: requests.Session, base_url: str, headers: dict,
             question: dict, mode: str) -> dict:
    payload = {"question": question["question"], "top_k": 5,
               "use_graph": True, "mode": mode}
    start = time.time()
    res = session.post(f"{base_url}/evaluate-query", json=payload,
                       headers=headers, timeout=180)
    elapsed = round(time.time() - start, 1)
    if res.status_code != 200:
        return {"error": f"HTTP {res.status_code}: {res.text[:120]}",
                "latency_s": elapsed}
    data = res.json()
    out = {k: data.get(k) for k in METRIC_KEYS}
    out["latency_s"] = elapsed
    out["answer_head"] = (data.get("answer") or "")[:90].replace("\n", " ")
    return out


def mean(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--out", default="benchmark_results_crag.json")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("API_KEY", "")
    headers = {"X-API-Key": api_key}
    session = requests.Session()

    results = {"answerable": {}, "unanswerable": {}}
    for bucket, questions in load_questions().items():
        for q in questions:
            row = {}
            for mode in ("hybrid", "crag"):
                out = run_mode(session, args.base_url, headers, q, mode)
                row[mode] = out
                status = "ERR " + str(out.get("error"))[:40] if "error" in out \
                    else f"faith={out.get('faithfulness')}"
                print(f"[{bucket[:4]}][{mode:>6}] q{q['id']}: {status} "
                      f"({out.get('latency_s')}s)")
            results[bucket][str(q["id"])] = {
                "question": q["question"],
                "type": q["question_type"],
                "hybrid": row["hybrid"],
                "crag": row["crag"],
            }

    all_rows = [r for bucket in results.values() for r in bucket.values()]

    def mode_mean(mode, key):
        return mean([r[mode].get(key) for r in all_rows
                     if "error" not in r[mode]])

    # Gate-compatible: "metrics" = candidate (crag) means over all questions.
    metrics = {k: mode_mean("crag", k) for k in METRIC_KEYS}
    hybrid_reference = {k: mode_mean("hybrid", k) for k in METRIC_KEYS}

    summary = {}
    for bucket, rows in results.items():
        values = list(rows.values())
        summary[bucket] = {
            f"{mode}_{k}": mean([r[mode].get(k) for r in values
                                 if "error" not in r[mode]])
            for mode in ("hybrid", "crag") for k in METRIC_KEYS
        }
        summary[bucket]["questions"] = len(values)

    artifact = {
        "version": 2,
        "kind": "mode_comparison",
        "last_updated": time.strftime("%Y-%m-%d"),
        # Candidate (CRAG) means over all questions - gate-consumable:
        "metrics": metrics,
        # Side-by-side reference for reporting:
        "hybrid_reference": hybrid_reference,
        "summary_by_bucket": summary,
        "per_question": results,
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("SUMMARY (mean over all questions)")
    print("=" * 62)
    for bucket in summary:
        print(f"\n{bucket} ({summary[bucket]['questions']} questions)")
        for k in METRIC_KEYS:
            h = summary[bucket][f"hybrid_{k}"]
            c = summary[bucket][f"crag_{k}"]
            print(f"  {k:>18}: hybrid={h:.4f}  crag={c:.4f}  "
                  f"delta={round(c - h, 4):+.4f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
