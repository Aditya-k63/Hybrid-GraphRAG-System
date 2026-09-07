import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

_retrieval_calls = defaultdict(int)
_total_queries = 0
_start_time = None


class RetrievalTracker:
    """Tracks retrieval calls to measure efficiency gains."""

    def __init__(self):
        self.call_counts = defaultdict(int)
        self.query_count = 0
        self.total_retrieval_calls = 0

    def record(self, retrieval_type: str, num_searches: int = 1):
        self.call_counts[retrieval_type] += 1
        self.total_retrieval_calls += num_searches
        self.query_count += 1

    def get_stats(self) -> dict:
        if self.query_count == 0:
            return {"queries": 0, "total_retrieval_calls": 0, "calls_per_query": 0}
        avg_calls = self.total_retrieval_calls / self.query_count
        baseline_calls = self.query_count * 3
        reduction = round((1 - avg_calls / baseline_calls) * 100, 1) if baseline_calls > 0 else 0.0
        return {
            "queries": self.query_count,
            "total_retrieval_calls": self.total_retrieval_calls,
            "calls_per_query": round(avg_calls, 2),
            "baseline_calls_if_no_classifier": baseline_calls,
            "retrieval_call_reduction_pct": reduction,
            "by_type": dict(self.call_counts),
        }

    def reset(self):
        self.call_counts.clear()
        self.query_count = 0
        self.total_retrieval_calls = 0


tracker = RetrievalTracker()


def get_tracker_stats() -> dict:
    return tracker.get_stats()


def get_latency_stats() -> dict:
    global _start_time
    return {
        "tracker_stats": tracker.get_stats(),
    }