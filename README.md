# Hybrid GraphRAG System

A production-oriented Retrieval-Augmented Generation (RAG) system that combines **semantic vector retrieval, BM25 keyword search, and knowledge-graph traversal** to improve retrieval quality for both straightforward and multi-hop questions.

The system ingests PDF documents, creates semantically meaningful chunks, generates embeddings, extracts entities and relationships, and stores the resulting knowledge across PostgreSQL/pgvector and Neo4j. At query time, it classifies the question, selects the appropriate retrieval strategy, fuses and reranks the retrieved evidence, and generates a grounded response with source references.

The project also includes API authentication, rate limiting, query caching, conversation memory, analytics, evaluation endpoints, Docker support, automated tests, and a GitHub Actions CI/CD pipeline.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │      PDF Upload      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Text Extraction    │
                         │  Semantic Chunking  │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌──────────────────┐             ┌──────────────────┐
          │   Embeddings     │             │ Entity / Relation│
          │ all-MiniLM-L6-v2 │             │    Extraction    │
          └────────┬─────────┘             └────────┬─────────┘
                   │                                │
                   ▼                                ▼
          ┌──────────────────┐             ┌──────────────────┐
          │ PostgreSQL       │             │      Neo4j        │
          │ + pgvector       │             │ Knowledge Graph  │
          └────────┬─────────┘             └────────┬─────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Query Classifier   │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          Vector Search        BM25 Search       Graph Search
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Reciprocal Rank     │
                         │ Fusion (RRF)        │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder       │
                         │ Reranking           │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ LLM Generation      │
                         │ + Source Context    │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Grounded Response   │
                         └─────────────────────┘
```

---

## Key Capabilities

### Multi-strategy retrieval

The system supports four retrieval modes:

- **Vector retrieval** — semantic similarity using pgvector.
- **BM25 retrieval** — lexical matching for exact names, terminology, dates, and keywords.
- **Graph retrieval** — relationship-based traversal through Neo4j.
- **Hybrid retrieval** — combines multiple retrieval signals using Reciprocal Rank Fusion (RRF).

A query classifier determines which retrieval strategy is appropriate for the incoming question. This allows the system to avoid unnecessary retrieval work while retaining the ability to handle relationship-heavy and multi-hop questions.

### Reranking

Retrieved candidates are passed through a cross-encoder reranker so that ranking is based on the relationship between the question and each candidate passage rather than retrieval score alone.

### Grounded generation

The generation layer receives the retrieved evidence and produces an answer grounded in that context. Source references are included so users can trace the response back to the retrieved document content.

### Evaluation

The project exposes evaluation APIs using RAGAS metrics, including:

- Faithfulness
- Answer relevance
- Context precision

When the live evaluation stack is unavailable or explicitly disabled, the application can use a deterministic semantic-similarity fallback. This fallback is a **proxy for evaluation only** and should not be interpreted as equivalent to RAGAS faithfulness.

---

## Recent Reliability & CI/CD Improvements

### Updating README for Recent Changes

The latest iteration focused on making the application more resilient in production-like environments and making CI reliable without depending on external LLM availability.

#### 1. Configurable live-LLM execution

The application now supports the `ENABLE_LIVE_LLM` setting.

```env
ENABLE_LIVE_LLM=true
```

In the normal application environment, live LLM calls remain enabled. In CI benchmark runs, the setting is disabled so the benchmark does not depend on Groq availability or consume external API quota.

This separation keeps the production architecture LLM-powered while allowing automated tests to validate retrieval, API behavior, database integration, and evaluation logic deterministically.

#### 2. More resilient query classification

The query classifier now handles imperfect LLM responses more defensively. Instead of assuming that every model response is valid JSON, the parser:

- removes common Markdown code fences;
- extracts JSON objects from surrounding text when necessary;
- validates the requested retrieval type; and
- falls back to a safe `hybrid` classification when parsing fails.

The classifier also retries transient Groq rate-limit responses with exponential backoff before falling back to the default retrieval strategy.

#### 3. LLM generation fallback and rate-limit handling

The generation layer now includes retry handling for HTTP 429 rate-limit responses.

If the live LLM is disabled or a live request ultimately fails, the application can return a deterministic response assembled from the retrieved context rather than causing the complete request to fail.

This is particularly useful for CI, local development, and degraded external-service conditions.

#### 4. Offline benchmark execution

The GitHub Actions benchmark no longer requires a live Groq request for every test case.

CI sets:

```env
ENABLE_LIVE_LLM=false
```

The benchmark therefore exercises the application using deterministic behavior. This avoids failures caused by external rate limits and makes CI results more reproducible.

Live LLM behavior can still be tested separately when API credentials and quota are available.

#### 5. Evaluation fallback hardening

The evaluation module was updated to tolerate differences in RAGAS result formats and to handle missing evaluation dependencies more safely.

The implementation now supports result normalization, batch result conversion, and a semantic-similarity fallback when live RAGAS evaluation cannot be performed.

The fallback is intentionally documented as a heuristic proxy rather than being presented as a replacement for true RAGAS faithfulness evaluation.

#### 6. PostgreSQL CI reliability

Database configuration was improved by making PostgreSQL SSL behavior configurable through `DB_SSLMODE` rather than forcing SSL in every environment.

For the local PostgreSQL service used by CI, SSL is disabled explicitly:

```env
DB_SSLMODE=disable
```

The PostgreSQL health check was also made explicit about the user and database instead of relying on the runner's default operating-system user.

#### 7. Safer graph retrieval

Graph retrieval is treated as an optional retrieval source. Failures in Neo4j graph operations are isolated so that the retrieval pipeline can fall back to other available retrieval methods instead of bringing down the complete query.

#### 8. GitHub Actions baseline updates

The CI workflow now explicitly grants the workflow permission to write repository contents when the benchmark baseline needs to be updated.

```yaml
permissions:
  contents: write
```

The checkout step also persists the GitHub-provided token, and the baseline update uses an explicit push to `main`.

This resolves the previous `403 Permission denied` failure that occurred when `github-actions[bot]` attempted to commit the generated benchmark baseline.

---

## Technology Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Language | Python 3.11+ |
| Vector Database | PostgreSQL + pgvector |
| Knowledge Graph | Neo4j |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Keyword Retrieval | BM25 |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder / ms-marco-MiniLM-L-6-v2 |
| LLM Provider | Groq |
| LLM Integration | LangChain / langchain-groq |
| Evaluation | RAGAS + semantic-similarity fallback |
| Entity Extraction | spaCy + LLM-based extraction |
| Frontend | Streamlit + built-in HTML UI |
| Containerization | Docker / Docker Compose |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## Project Structure

```text
Hybrid-GraphRAG-System/
├── app/
│   ├── main.py                  # FastAPI application and routes
│   ├── config.py                # Environment configuration
│   ├── database.py              # PostgreSQL connection management
│   ├── graph.py                 # Neo4j operations
│   ├── auth.py                  # API-key authentication and rate limiting
│   ├── models.py                # Pydantic request/response models
│   │
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PDF text extraction
│   │   ├── chunker.py           # Semantic chunking
│   │   ├── embedder.py          # Embedding generation
│   │   └── entity_extractor.py  # Entity and relationship extraction
│   │
│   ├── retrieval/
│   │   ├── vector_search.py     # pgvector semantic retrieval
│   │   ├── bm25_search.py       # BM25 keyword retrieval
│   │   ├── graph_search.py      # Neo4j graph retrieval
│   │   ├── hybrid.py            # Hybrid retrieval and RRF
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── query_classifier.py  # Retrieval strategy classification
│   │
│   ├── generation/
│   │   └── llm.py               # LLM generation and fallbacks
│   │
│   ├── memory/
│   │   └── conversation.py      # Session-based conversation memory
│   │
│   └── evaluation/
│       └── ragas.py             # RAGAS and fallback evaluation
│
├── frontend/
│   └── app.py                   # Streamlit frontend
│
├── static/
│   └── index.html               # Built-in web interface
│
├── scripts/
│   └── seed_benchmark_data.py   # Benchmark database seeding
│
├── tests/
│   ├── test_ragas.py
│   ├── test_benchmark.py
│   └── ...
│
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL with the `pgvector` extension
- Neo4j 5+
- Groq API key for live LLM functionality
- Docker and Docker Compose (optional)

### Installation

```bash
git clone https://github.com/Aditya-k63/Hybrid-GraphRAG-System.git
cd Hybrid-GraphRAG-System

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# PostgreSQL
DB_NAME=graphrag
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_SSLMODE=require

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# LLM
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-20b
ENABLE_LIVE_LLM=true

# Application authentication
API_KEY=change-me-in-production
```

For a local PostgreSQL instance without SSL, use:

```env
DB_SSLMODE=disable
```

### Database Setup

Create the PostgreSQL database and enable pgvector:

```sql
CREATE DATABASE graphrag;

-- Connect to graphrag, then:
CREATE EXTENSION IF NOT EXISTS vector;
```

Create a Neo4j instance and configure its connection details in `.env`. Required graph constraints are handled by the application during startup.

---

## Running the Application

### FastAPI

```bash
uvicorn app.main:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

### Streamlit

```bash
streamlit run frontend/app.py
```

Streamlit UI:

```text
http://localhost:8501
```

The built-in web UI is available at:

```text
http://localhost:8000
```

### Docker Compose

```bash
docker-compose up --build
```

Docker Compose starts the application services together with the required infrastructure.

---

## API Reference

All endpoints except `/` and `/health` require the `X-API-Key` header.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Built-in web interface |
| GET | `/health` | Service health/status |
| GET | `/documents` | List ingested documents |
| POST | `/upload` | Upload and process a PDF |
| POST | `/query` | Run a retrieval + generation query |
| POST | `/evaluate-query` | Query with evaluation scores |
| POST | `/evaluate-batch` | Batch evaluation |
| GET | `/analytics` | Search analytics |
| GET | `/analytics/retrieval-stats` | Retrieval efficiency statistics |
| POST | `/memory/{id}/clear` | Clear conversation memory |
| POST | `/cache/clear` | Clear query cache |

---

## Retrieval Pipeline

The system is designed around the idea that no single retrieval method is optimal for every question.

### Vector Search

Embeddings represent document chunks in a semantic vector space. pgvector is used to retrieve passages that are conceptually similar to the query, even when exact keywords do not match.

### BM25 Search

BM25 provides lexical retrieval. It is particularly useful when the query contains exact names, technical terms, identifiers, dates, or phrases that semantic retrieval may underweight.

### Graph Search

Entities and relationships extracted from documents are represented in Neo4j. Graph traversal can follow relationships across multiple entities and documents, making it useful for relationship-heavy questions.

### Reciprocal Rank Fusion

Results from the available retrieval methods are combined using Reciprocal Rank Fusion. Instead of treating the score from one retrieval system as universally comparable to another, RRF combines their rankings into a unified candidate list.

### Cross-Encoder Reranking

The candidate passages are then evaluated by a cross-encoder using the query and passage together. This provides a final relevance ranking before context is passed to the generation layer.

---

## LLM Reliability Model

Live LLM requests are protected against common transient failures.

```text
Application Request
        │
        ▼
   Live LLM enabled?
      │        │
     Yes       No
      │        │
      ▼        ▼
  Groq call   Deterministic fallback
      │
      ▼
   429 / error?
      │
   ┌──┴──┐
  retry  fallback
```

The classifier and generation components use retry/backoff handling for rate-limit responses. If a request still cannot be completed, the application uses a safe fallback path where supported.

A single Groq API key can be used with different configured models, but changing models does not create independent account-level quota. Model routing and deterministic fallbacks are therefore complementary reliability mechanisms rather than a replacement for quota management.

---

## Evaluation Strategy

The project supports two evaluation paths.

### Live RAGAS evaluation

When live LLM evaluation is enabled and the required dependencies are available, RAGAS evaluates:

- **Faithfulness** — whether the generated answer is supported by the retrieved context.
- **Answer relevance** — whether the answer addresses the question.
- **Context precision** — whether retrieved context is useful for the answer.

### Deterministic fallback evaluation

When live evaluation is disabled, such as in CI benchmarks, the system uses embedding-based semantic similarity as a deterministic proxy.

This distinction is important: **semantic similarity is not the same measurement as RAGAS faithfulness**. The fallback exists to keep automated infrastructure tests reproducible and independent of external LLM availability.

---

## Testing

Run the complete test suite with:

```bash
python -m pytest tests/ -v
```

The test suite covers application behavior, ingestion, retrieval logic, evaluation behavior, and benchmark execution.

The benchmark execution is structured so the expensive benchmark run is performed once for the test session rather than once for every individual metric assertion.

---

## CI/CD

The GitHub Actions pipeline is designed to validate the application and maintain benchmark artifacts automatically.

High-level workflow:

```text
Push / Pull Request
        │
        ▼
   Build & Tests
        │
        ▼
 Benchmark Environment
 PostgreSQL + pgvector
       + Neo4j
        │
        ▼
 Offline Benchmark Mode
 ENABLE_LIVE_LLM=false
        │
        ▼
 Benchmark Results
        │
        ├──────────────► Docker Build / Push
        │
        ▼
 Benchmark Baseline Update
        │
        ▼
 Commit generated baseline
```

The benchmark environment intentionally avoids live Groq requests. This prevents external rate limits from determining whether infrastructure and retrieval tests pass.

The workflow also has explicit repository-content write permission for the automated baseline update:

```yaml
permissions:
  contents: write
```

---

## Production vs CI Configuration

| Concern | Local / Production | CI Benchmark |
|---|---|---|
| Live LLM | Enabled by default | Disabled |
| Groq API dependency | Required for live generation/evaluation | Not required for benchmark execution |
| Generation | Groq LLM | Deterministic context fallback |
| Query classification | LLM-based with fallback | Deterministic hybrid mode |
| Evaluation | RAGAS when available | Semantic-similarity fallback |
| PostgreSQL SSL | Configurable | Disabled for local CI service |
| External API rate-limit impact | Possible and handled with retry/backoff | Avoided |

This separation allows CI to validate the software system without making the CI result dependent on an external model provider.

---

## Design Principles

### Reliability over single-provider dependency

External LLM calls are treated as failure-prone dependencies. Retry logic, safe defaults, and deterministic fallbacks prevent transient provider failures from unnecessarily breaking the application.

### Retrieval diversity

Semantic, lexical, and graph retrieval solve different information-retrieval problems. Combining them provides broader coverage than relying on a single retrieval signal.

### Reproducible CI

Automated tests should not fail simply because an external API is rate-limited. CI therefore uses deterministic LLM-disabled execution for benchmark validation.

### Clear evaluation semantics

The project distinguishes true RAGAS evaluation from heuristic semantic similarity instead of treating the two as interchangeable metrics.

### Production-oriented engineering

The system includes authentication, rate limiting, caching, analytics, containerization, automated testing, configuration management, and CI/CD alongside the core RAG pipeline.

---

## Project Status

| Capability | Status |
|---|---|
| PDF ingestion | Implemented |
| Semantic chunking | Implemented |
| Embedding generation | Implemented |
| PostgreSQL + pgvector | Implemented |
| Neo4j knowledge graph | Implemented |
| BM25 retrieval | Implemented |
| Hybrid retrieval / RRF | Implemented |
| Cross-encoder reranking | Implemented |
| Query classification | Implemented |
| Multi-hop graph retrieval | Implemented |
| LLM generation | Implemented |
| Source citations | Implemented |
| Conversation memory | Implemented |
| RAGAS evaluation | Implemented |
| Evaluation fallback | Implemented |
| API authentication | Implemented |
| Rate limiting | Implemented |
| Query caching | Implemented |
| Search analytics | Implemented |
| Docker Compose | Implemented |
| Automated tests | Implemented |
| GitHub Actions CI/CD | Implemented |

---

## Author

**Aditya Kumar**

GitHub: https://github.com/Aditya-k63

---

## License

Add the project's license here when a license is formally selected for the repository.
