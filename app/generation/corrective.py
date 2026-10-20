"""Corrective RAG (CRAG) layer for the Hybrid-GraphRAG system.

Flow:
    retrieve (existing hybrid pipeline) -> grade (LLM: yes / no)
        |- yes -> generate from the retrieved chunks
        `- no  -> web_fallback (DuckDuckGo) -> generate

The default /query path is untouched: this module runs only when a
request is sent with mode "crag".  When ENABLE_LIVE_LLM is disabled
(CI / offline benchmarks) the grading step returns a deterministic
"yes" so benchmark runs stay reproducible and make no external calls.
"""
from typing import TypedDict

from app.config import settings

GRADE_MESSAGE = """You are a retrieval quality grader for a RAG system.
Decide whether the context below is sufficient to answer the user's question.

Reply with exactly one word: yes or no.
Do not add punctuation or explanation.

Question:
{question}

Context:
{context}
"""


class CorrectiveState(TypedDict):
    query: str
    top_k: int
    use_graph: bool
    history: list
    chunks: list
    retrieval_type: str
    grade: str
    used_web: bool
    answer: str


def _retrieve_node(state: CorrectiveState) -> dict:
    """Run the existing hybrid retrieval pipeline (classifier + RRF + rerank)."""
    from app.retrieval.vector_search import vector_search
    from app.retrieval.bm25_search import bm25_search
    from app.retrieval.graph_search import graph_retrieve
    from app.retrieval.hybrid import reciprocal_rank_fusion
    from app.retrieval.reranker import rerank
    from app.retrieval.query_classifier import classify_query

    query = state["query"]
    classification = classify_query(query)
    retrieval_type = classification["type"]

    result_lists: list[list[dict]] = []
    if retrieval_type in ("vector", "hybrid"):
        result_lists.append(vector_search(query, top_k=settings.VECTOR_TOP_K))
    if retrieval_type in ("vector", "hybrid", "graph"):
        result_lists.append(bm25_search(query, top_k=settings.BM25_TOP_K))
    if state["use_graph"] and retrieval_type in ("graph", "hybrid"):
        graph_results = graph_retrieve(query, max_hops=settings.MAX_HOPS)
        if graph_results:
            result_lists.append(graph_results)
    if not result_lists:
        result_lists.append(vector_search(query, top_k=state["top_k"]))

    fused = reciprocal_rank_fusion(result_lists, k=settings.RRF_K)
    chunks = rerank(query, fused[:20], top_k=state["top_k"])
    return {"chunks": chunks, "retrieval_type": retrieval_type}


def _grade_node(state: CorrectiveState) -> dict:
    """Grade whether the retrieved chunks can answer the query (yes / no)."""
    chunks = state["chunks"]
    context = "\n\n".join(
        c.get("content", "") for c in chunks
    )[: settings.VERIFY_MAX_CONTEXT_CHARS]

    if not context.strip():
        return {"grade": "no"}

    if not settings.ENABLE_LIVE_LLM:
        # CI / offline benchmark path: deterministic grade, no external calls.
        return {"grade": "yes"}

    from app.generation.llm import get_groq
    from app.utils.retry import with_retry

    @with_retry(max_attempts=2, base_delay=2.0)
    def _ask_grader() -> str:
        client = get_groq()
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": GRADE_MESSAGE.format(
                    question=state["query"], context=context),
            }],
            temperature=0.0,
            max_tokens=3,
        )
        return (response.choices[0].message.content or "").strip().lower()

    try:
        text = _ask_grader()
    except Exception:
        # Grader unavailable: treat as "no" so the web fallback gets a chance.
        text = "no"
    return {"grade": "yes" if text.startswith("yes") else "no"}


def _route_after_grade(state: CorrectiveState) -> str:
    """Conditional edge: generate locally, or correct via web fallback."""
    if state["grade"] == "yes" and state["chunks"]:
        return "generate"
    return "web_fallback"


def _web_fallback_node(state: CorrectiveState) -> dict:
    """Replace weak local context with DuckDuckGo web results."""
    results = []
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(
                state["query"], max_results=settings.CRAG_WEB_RESULTS))
    except Exception:
        results = []

    web_chunks: list[dict] = []
    for result in results:
        title = result.get("title", "")
        body = result.get("body", "")
        href = result.get("href") or result.get("link", "")
        if title and body:
            content = f"{title}: {body}"
        else:
            content = body or title
        if not content:
            continue
        source = f"web: {href}" if href else "web"
        web_chunks.append({"content": content, "meta": {"source": source}})
    return {"chunks": web_chunks, "used_web": bool(web_chunks)}


def _generate_node(state: CorrectiveState) -> dict:
    """Generate the final grounded answer from the chosen context."""
    from app.generation.llm import generate_answer

    answer = generate_answer(
        state["query"], state["chunks"], state.get("history") or [])
    return {"answer": answer}


def _build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(CorrectiveState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("grade", _grade_node)
    graph.add_node("web_fallback", _web_fallback_node)
    graph.add_node("generate", _generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", _route_after_grade, {
        "generate": "generate",
        "web_fallback": "web_fallback",
    })
    graph.add_edge("web_fallback", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_compiled = None


def run_corrective(
    query: str,
    top_k: int = 5,
    use_graph: bool = True,
    conversation_history: list | None = None,
) -> dict:
    """Entry point used by /query when mode == "crag"."""
    global _compiled
    if _compiled is None:
        _compiled = _build_graph()

    state = _compiled.invoke({
        "query": query,
        "top_k": top_k,
        "use_graph": use_graph,
        "history": conversation_history or [],
        "chunks": [],
        "retrieval_type": "",
        "grade": "",
        "used_web": False,
        "answer": "",
    })
    return {
        "answer": state["answer"],
        "chunks": state["chunks"],
        "retrieval_type": state.get("retrieval_type") or "crag",
        "grade": state.get("grade", ""),
        "used_web": state.get("used_web", False),
    }
