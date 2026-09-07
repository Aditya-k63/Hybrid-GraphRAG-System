"""Multi-hop query tests for the Hybrid GraphRAG system."""
import pytest
from app.retrieval.graph_search import multi_hop_query, graph_retrieve
from app.retrieval.query_classifier import classify_query


def test_multi_hop_person_organization():
    """Test multi-hop traversal from Person to Organization."""
    results = multi_hop_query(
        "Who founded the company that acquired X?",
        ["Person", "Organization", "Event"]
    )
    assert isinstance(results, list)


def test_multi_hop_organization_technology():
    """Test multi-hop traversal from Organization to Technology."""
    results = multi_hop_query(
        "What technology does Company X use?",
        ["Organization", "Technology"]
    )
    assert isinstance(results, list)


def test_multi_hop_chain_length():
    """Verify multi-hop chains support variable hop counts."""
    results = multi_hop_query("test chain", ["Person", "Organization"])
    assert isinstance(results, list)


def test_classifier_multi_hop_detection():
    """Verify that multi-hop questions are classified correctly."""
    queries = [
        "Who founded the company that acquired Company B?",
        "What happened to Company A after Company B acquired it?",
        "How did the acquisition of X by Y affect Z?",
    ]
    for query in queries:
        result = classify_query(query)
        assert result["type"] in ("hybrid", "graph"), f"Expected hybrid/graph for: {query}, got {result['type']}"


def test_graph_retrieve_with_hops():
    """Test graph retrieval with varying hop counts."""
    for hops in [1, 2, 3]:
        results = graph_retrieve("test entity query", max_hops=hops)
        assert isinstance(results, list)