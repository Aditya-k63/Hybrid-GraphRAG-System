# Hybrid GraphRAG System

Most RAG projects just do vector search. This one combines three retrieval methods — vector search, keyword search, and a knowledge graph — to answer questions that none of those approaches could handle alone.

You upload a PDF. The system extracts text, chunks it, generates embeddings, and stores everything in PostgreSQL with pgvector. It also extracts entities and relationships using an LLM and stores them in a Neo4j knowledge graph. When you ask a question, a query classifier decides which retrieval strategy to use, merges results with Reciprocal Rank Fusion, reranks them with a cross-encoder, and generates a grounded answer with source citations.

---

## How it works

```
Upload PDF
    ↓
Text extraction → Semantic chunking → Embedding generation
    ↓                                    ↓
PostgreSQL (pgvector)            Entity extraction → Neo4j graph
    ↓                                    ↓
Ask a question
    ↓
Query classifier picks strategy:
├─ Vector: semantic similarity
├─ Graph: entity relationship traversal
├─ BM25: keyword matching
└─ Hybrid: combines all three
    ↓
Reciprocal Rank Fusion merges results
    ↓
Cross-encoder reranker picks best chunks
    ↓
LLM generates answer with citations
```

---

## What's inside

| Feature | Status |
|---|---|
| PDF ingestion | Done |
| Semantic chunking | Done |
| Embedding generation (all-MiniLM-L6-v2) | Done |
| PostgreSQL + pgvector | Done |
| Neo4j knowledge graph | Done |
| BM25 keyword search | Done |
| Hybrid retrieval (RRF) | Done |
| Cross-encoder reranker | Done |
| Query classifier | Done |
| Multi-hop graph traversal | Done |
| LLM generation (Groq / llama-3.1) | Done |
| Source citations | Done |
| Conversation memory | Done |
| RAGAS evaluation | Done |
| API key authentication | Done |
| Rate limiting | Done |
| Query caching | Done |
| Search analytics | Done |
| Docker Compose | Done |
| Tests | Done |
| CI/CD (GitHub Actions) | Done |

---

## Tech stack

| Layer | Tool |
|---|---|
| Vector DB | pgvector (PostgreSQL) |
| Knowledge Graph | Neo4j |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| LLM | Groq (llama-3.1-8b-instant) |
| Backend | FastAPI |
| Frontend | Streamlit + built-in HTML UI |
| Containerization | Docker Compose |

---

## Project structure

```
Hybrid-GraphRAG-System/
├── app/
│   ├── main.py                  # FastAPI app + all routes
│   ├── config.py                # Environment settings
│   ├── database.py              # PostgreSQL connection pool
│   ├── graph.py                 # Neo4j operations
│   ├── auth.py                  # API key auth + rate limiting
│   ├── models.py                # Pydantic schemas
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PDF text extraction
│   │   ├── chunker.py           # Semantic chunking
│   │   ├── embedder.py          # Embedding generation
│   │   └── entity_extractor.py  # LLM-based NER
│   ├── retrieval/
│   │   ├── vector_search.py     # pgvector semantic search
│   │   ├── bm25_search.py       # BM25 keyword search
│   │   ├── graph_search.py      # Neo4j traversal
│   │   ├── hybrid.py            # Reciprocal rank fusion
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── query_classifier.py  # Routes queries to retrieval type
│   ├── generation/
│   │   └── llm.py               # Groq LLM integration
│   ├── memory/
│   │   └── conversation.py      # Session-based chat history
│   └── evaluation/
│       └── ragas.py             # Faithfulness, relevance, precision
├── tests/
├── frontend/
│   └── app.py                   # Streamlit UI
├── static/
│   └── index.html               # Built-in web UI
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .github/workflows/
    └── ci.yml
```

---

## Getting started

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- Neo4j 5+
- Groq API key (free at console.groq.com)

### Install

```bash
git clone https://github.com/Aditya-k63/Enterprise-Hybrid-GraphRAG.git
cd Enterprise-Hybrid-GraphRAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create `.env` in root:

```env
DB_NAME=graphrag
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

GROQ_API_KEY=your_groq_api_key
API_KEY=change-me-in-production
```

### Database setup

**PostgreSQL:**
```sql
CREATE DATABASE graphrag;
CREATE EXTENSION IF NOT EXISTS vector;
```

**Neo4j:** Create the instance, constraints are created automatically on first run.

### Run locally

```bash
# Terminal 1 — API
uvicorn app.main:app --reload

# Terminal 2 — Streamlit (optional)
streamlit run frontend/app.py
```

- API docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501
- Built-in UI: http://localhost:8000

### Docker Compose

```bash
docker-compose up --build
```

This starts Neo4j, the API, and Streamlit together.

---

## API endpoints

All endpoints except `/` and `/health` require `X-API-Key` header.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | System status |
| GET | `/documents` | List ingested PDFs |
| POST | `/upload` | Upload PDF |
| POST | `/query` | Ask a question |
| POST | `/evaluate-query` | Ask + get quality scores |
| GET | `/analytics` | Search analytics |
| POST | `/memory/{id}/clear` | Clear session memory |
| POST | `/cache/clear` | Clear query cache |

---

## How retrieval works

Most RAG projects do one thing: vector search. That works until someone asks "Who founded the company that acquired X?" — vector search can't follow that chain.

This system runs up to three searches in parallel:

1. **Vector search** — finds chunks with similar meaning (pgvector cosine similarity)
2. **BM25 search** — finds chunks with matching keywords, names, dates
3. **Graph traversal** — follows entity relationships in Neo4j (Person → founded → Company → acquired → Company)

Results are merged using Reciprocal Rank Fusion, which combines rankings from different sources into a single score. Then a cross-encoder reranker looks at each (question, chunk) pair and picks the best ones.

A query classifier decides which searches to run — simple questions skip the graph, entity-heavy questions prioritize it.

---

## Evaluation

Every `/evaluate-query` response includes:

- **Faithfulness** — is the answer grounded in the retrieved chunks?
- **Answer relevance** — does it actually answer the question?
- **Context precision** — were the retrieved chunks useful?

Scores are logged to PostgreSQL for tracking over time.

---

## Tests

```bash
python -m pytest tests/ -v
```

Covers API endpoints, ingestion pipeline, and retrieval fusion logic.

---

## CI/CD

GitHub Actions workflow:
1. Runs tests on every push
2. Builds Docker image
3. Pushes to Docker Hub as `yourusername/graphrag:latest`

---

## Why hybrid?

| Question type | Vector alone | Graph alone | Hybrid |
|---|---|---|---|
| "What is machine learning?" | Works | Fails | Works |
| "Who founded Google?" | Might miss | Works | Works |
| "How did Company A's acquisition affect Company B?" | Fails | Works | Works |
| "Explain quantum entanglement" | Works | Fails | Works |

Hybrid covers all cases. The query classifier picks the right strategy automatically.

---

> Built by [Aditya Kumar](https://github.com/Aditya-k63)
