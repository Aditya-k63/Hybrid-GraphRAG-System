import os
import json
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str | None = None
    use_graph: bool = True

from app.retrieval.tracker import tracker, get_tracker_stats

STATIC_DIR = Path(__file__).parent.parent / "static"
query_cache = {}


@asynccontextmanager
async def lifespan(app):
    logger.info("Starting Enterprise Hybrid GraphRAG...")
    try:
        from app.database import init_db
        init_db()
    except Exception as e:
        logger.warning(f"DB init skipped: {e}")
    try:
        from app.graph import init_graph
        init_graph()
    except Exception as e:
        logger.warning(f"Neo4j init skipped: {e}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Enterprise Hybrid GraphRAG",
    description="Hybrid retrieval combining vector search, BM25, and Neo4j knowledge graph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    from app.auth import rate_limiter
    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(rate_limiter.remaining(client_ip))
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed}ms)")
    return response


@app.get("/")
def root():
    index_path = STATIC_DIR / "index.html"
    try:
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read index.html: {e}")
    return HTMLResponse(content="<h1>Enterprise Hybrid GraphRAG</h1><p><a href='/docs'>API Docs</a></p>")


@app.get("/health")
def health():
    from app.database import db_available
    pg_ok = db_available()
    neo4j_ok = False
    try:
        from app.graph import get_driver
        d = get_driver()
        if d:
            d.verify_connectivity()
            neo4j_ok = True
    except Exception:
        pass
    status = "healthy" if pg_ok else "degraded"
    return {
        "status": status,
        "postgres": "connected" if pg_ok else "disconnected",
        "neo4j": "connected" if neo4j_ok else "disconnected",
        "cache_size": len(query_cache),
    }


@app.get("/documents")
def list_documents(request: Request):
    from app.auth import check_api_key
    from app.models import DocumentInfo
    from app.database import get_connection, release_connection, db_available
    from app.graph import get_document_entities

    check_api_key(request)
    if not db_available():
        raise HTTPException(status_code=503, detail="Database not available")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT meta->>'source' AS source, COUNT(*) AS chunks
            FROM document_sections
            WHERE meta IS NOT NULL
            GROUP BY meta->>'source'
            ORDER BY source;
        """)
        rows = cur.fetchall()
        cur.close()

        results = []
        for row in rows:
            filename = row[0]
            chunk_count = row[1]
            try:
                entities = get_document_entities(filename)
            except Exception:
                entities = []
            results.append(DocumentInfo(
                filename=filename,
                chunks=chunk_count,
                entities=len(entities),
                uploaded_at=None,
            ))
        return results
    finally:
        release_connection(conn)


@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...), force: bool = False):
    from app.auth import check_api_key
    from app.models import UploadResponse
    from app.database import get_connection, release_connection, db_available
    from app.config import settings
    from app.graph import store_entities, delete_document_graph
    from app.ingestion.pdf_parser import extract_text_from_pdf
    from app.ingestion.chunker import chunk_text
    from app.ingestion.embedder import embed_chunks
    from app.ingestion.entity_extractor import extract_entities

    check_api_key(request)

    content = await file.read()
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.MAX_FILE_SIZE // 1024 // 1024}MB")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if not db_available():
        raise HTTPException(status_code=503, detail="Database not available. Cannot upload documents.")

    if not force:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_sections WHERE meta->>'source' = %s", (file.filename,))
            count = cur.fetchone()[0]
            cur.close()
            if count > 0:
                raise HTTPException(status_code=409, detail=f"'{file.filename}' already ingested. Use ?force=true to re-upload.")
        finally:
            release_connection(conn)

    if force:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM document_sections WHERE meta->>'source' = %s", (file.filename,))
            conn.commit()
            cur.close()
        finally:
            release_connection(conn)
        try:
            delete_document_graph(file.filename)
        except Exception:
            pass

    text = extract_text_from_pdf(content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    chunks = chunk_text(text)
    if len(chunks) > settings.MAX_CHUNKS:
        raise HTTPException(status_code=400, detail=f"PDF too large — {len(chunks)} chunks, max {settings.MAX_CHUNKS}")

    embeddings = embed_chunks(chunks)

    conn = get_connection()
    inserted = 0
    try:
        cur = conn.cursor()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {"source": file.filename, "chunk_index": i}
            cur.execute(
                "INSERT INTO document_sections (content, meta, embedding) VALUES (%s, %s, %s)",
                (chunk, json.dumps(metadata), embedding),
            )
            inserted += 1
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_connection(conn)

    entities_data = {"entities": [], "relationships": []}
    try:
        full_text = " ".join(chunks[:20])
        entities_data = extract_entities(full_text)
        if entities_data["entities"]:
            store_entities(file.filename, entities_data["entities"], entities_data["relationships"])
    except Exception as e:
        logger.error(f"Entity extraction failed for '{file.filename}': {e}")

    query_cache.clear()

    return UploadResponse(
        filename=file.filename,
        chunks_inserted=inserted,
        entities_extracted=len(entities_data["entities"]),
        relationships_found=len(entities_data["relationships"]),
        message=f"Successfully ingested '{file.filename}'",
    )


@app.post("/query")
async def query_endpoint(request: QueryRequest, raw_request: Request):
    from app.auth import check_api_key
    from app.models import QueryResponse
    from app.database import db_available
    from app.config import settings
    from app.retrieval.vector_search import vector_search
    from app.retrieval.bm25_search import bm25_search
    from app.retrieval.graph_search import graph_retrieve
    from app.retrieval.hybrid import reciprocal_rank_fusion
    from app.retrieval.reranker import rerank
    from app.retrieval.query_classifier import classify_query
    from app.generation.llm import generate_answer
    from app.memory.conversation import memory

    check_api_key(raw_request)

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cache_key = f"{request.question.strip().lower()}::{request.top_k}::{request.use_graph}"
    if cache_key in query_cache:
        return QueryResponse(**query_cache[cache_key])

    classification = classify_query(request.question)
    retrieval_type = classification["type"]
    logger.info(f"Query classified as: {retrieval_type} ({classification['reason']})")

    all_results = []

    retrieval_calls = 0
    if retrieval_type in ("vector", "hybrid"):
        vec_results = vector_search(request.question, top_k=settings.VECTOR_TOP_K)
        all_results.append(vec_results)
        retrieval_calls += 1

    if retrieval_type in ("vector", "hybrid", "graph"):
        bm25_results = bm25_search(request.question, top_k=settings.BM25_TOP_K)
        all_results.append(bm25_results)
        retrieval_calls += 1

    if request.use_graph and retrieval_type in ("graph", "hybrid"):
        graph_results = graph_retrieve(request.question, max_hops=settings.MAX_HOPS)
        if graph_results:
            all_results.append(graph_results)
        retrieval_calls += 1

    if not all_results:
        vec_results = vector_search(request.question, top_k=request.top_k)
        all_results.append(vec_results)
        retrieval_calls += 1

    tracker.record(retrieval_type, retrieval_calls)

    fused = reciprocal_rank_fusion(all_results, k=settings.RRF_K)
    reranked = rerank(request.question, fused[:20], top_k=request.top_k)

    start = time.time()
    chunks = reranked
    latency_ms = (time.time() - start) * 1000

    history = []
    if request.session_id:
        history = memory.get_history(request.session_id)

    answer = generate_answer(request.question, chunks, history)

    if request.session_id:
        memory.add_message(request.session_id, "user", request.question)
        memory.add_message(request.session_id, "assistant", answer)

    sources = []
    for chunk in chunks:
        meta = chunk.get("meta")
        if isinstance(meta, dict) and meta.get("source"):
            if meta["source"] not in sources:
                sources.append(meta["source"])

    result = {
        "question": request.question,
        "answer": answer,
        "chunks_used": len(chunks),
        "retrieval_type": retrieval_type,
        "sources": sources,
    }
    query_cache[cache_key] = result

    if db_available():
        try:
            from app.database import get_connection, release_connection
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO search_analytics (question, retrieval_type, chunks_used, latency_ms) VALUES (%s, %s, %s, %s)",
                (request.question, retrieval_type, len(chunks), latency_ms),
            )
            conn.commit()
            cur.close()
            release_connection(conn)
        except Exception:
            pass

    result["retrieval_calls"] = retrieval_calls
    result["latency_ms"] = round(latency_ms, 1)
    return QueryResponse(**result)


@app.post("/evaluate-query")
async def evaluate_query(request: QueryRequest, raw_request: Request):
    from app.auth import check_api_key
    from app.models import EvaluatedQueryResponse
    from app.config import settings
    from app.retrieval.vector_search import vector_search
    from app.retrieval.bm25_search import bm25_search
    from app.retrieval.graph_search import graph_retrieve
    from app.retrieval.hybrid import reciprocal_rank_fusion
    from app.retrieval.reranker import rerank
    from app.retrieval.query_classifier import classify_query
    from app.generation.llm import generate_answer
    from app.evaluation.ragas import evaluate

    check_api_key(raw_request)

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    classification = classify_query(request.question)
    retrieval_type = classification["type"]

    eval_retrieval_calls = 0
    if retrieval_type in ("vector", "hybrid"):
        all_results.append(vector_search(request.question, top_k=settings.VECTOR_TOP_K))
        eval_retrieval_calls += 1
    if retrieval_type in ("vector", "hybrid", "graph"):
        all_results.append(bm25_search(request.question, top_k=settings.BM25_TOP_K))
        eval_retrieval_calls += 1
    if request.use_graph and retrieval_type in ("graph", "hybrid"):
        gr = graph_retrieve(request.question, max_hops=settings.MAX_HOPS)
        if gr:
            all_results.append(gr)
        eval_retrieval_calls += 1
    if not all_results:
        all_results.append(vector_search(request.question, top_k=request.top_k))
        eval_retrieval_calls += 1
    tracker.record(retrieval_type, eval_retrieval_calls)

    fused = reciprocal_rank_fusion(all_results, k=settings.RRF_K)
    chunks = rerank(request.question, fused[:20], top_k=request.top_k)

    answer = generate_answer(request.question, chunks)
    chunks_text = [c["content"] for c in chunks]
    metrics = evaluate(request.question, answer, chunks_text)

    return EvaluatedQueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=len(chunks),
        retrieval_type=retrieval_type,
        **metrics,
    )


class BatchEvalRequest(BaseModel):
    questions: list[str]
    top_k: int = 5
    use_graph: bool = True


@app.post("/evaluate-batch")
async def evaluate_batch_endpoint(request: BatchEvalRequest, raw_request: Request):
    from app.auth import check_api_key
    from app.models import EvaluatedQueryResponse
    from app.config import settings
    from app.retrieval.vector_search import vector_search
    from app.retrieval.bm25_search import bm25_search
    from app.retrieval.graph_search import graph_retrieve
    from app.retrieval.hybrid import reciprocal_rank_fusion
    from app.retrieval.reranker import rerank
    from app.retrieval.query_classifier import classify_query
    from app.generation.llm import generate_answer
    from app.evaluation.ragas import evaluate_batch

    check_api_key(raw_request)

    questions = request.questions
    if not questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    answers = []
    contexts_list = []

    for q in questions:
        classification = classify_query(q)
        retrieval_type = classification["type"]
        all_results = []
        if retrieval_type in ("vector", "hybrid"):
            all_results.append(vector_search(q, top_k=settings.VECTOR_TOP_K))
        if retrieval_type in ("vector", "hybrid", "graph"):
            all_results.append(bm25_search(q, top_k=settings.BM25_TOP_K))
        if request.use_graph and retrieval_type in ("graph", "hybrid"):
            gr = graph_retrieve(q)
            if gr:
                all_results.append(gr)
        if not all_results:
            all_results.append(vector_search(q, top_k=request.top_k))

        fused = reciprocal_rank_fusion(all_results, k=settings.RRF_K)
        chunks = rerank(q, fused[:20], top_k=request.top_k)
        answer = generate_answer(q, chunks)
        answers.append(answer)
        contexts_list.append([c["content"] for c in chunks])

    metrics = evaluate_batch(questions, answers, contexts_list)
    return metrics


@app.post("/memory/{session_id}/clear")
def clear_memory(session_id: str, request: Request):
    from app.auth import check_api_key
    from app.memory.conversation import memory
    check_api_key(request)
    memory.clear(session_id)
    return {"message": f"Memory cleared for session {session_id}"}


@app.post("/cache/clear")
def clear_cache(request: Request):
    from app.auth import check_api_key
    check_api_key(request)
    count = len(query_cache)
    query_cache.clear()
    return {"message": f"Cache cleared. {count} entries removed."}


@app.get("/analytics/retrieval-stats")
def get_retrieval_stats(request: Request):
    from app.auth import check_api_key
    check_api_key(request)
    return get_tracker_stats()


@app.get("/analytics")
def get_analytics(request: Request):
    from app.auth import check_api_key
    from app.database import get_connection, release_connection, db_available

    check_api_key(request)
    if not db_available():
        raise HTTPException(status_code=503, detail="Database not available")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                retrieval_type,
                COUNT(*) as count
            FROM search_analytics
            GROUP BY retrieval_type
        """)
        rows = cur.fetchall()
        cur.close()

        total = sum(r[1] for r in rows)
        by_type = {r[0]: {"count": r[1], "percentage": round(r[1] / total * 100, 1) if total else 0} for r in rows}

        cur = conn.cursor()
        cur.execute("SELECT AVG(latency_ms) FROM search_analytics")
        avg_latency = cur.fetchone()[0]
        cur.close()

        return {
            "total_queries": total,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0,
            "by_retrieval_type": by_type,
        }
    finally:
        release_connection(conn)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
