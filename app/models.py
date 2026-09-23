from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str | None = None
    use_graph: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    retrieval_type: str
    sources: list[str]
    verified: bool | None = None
    confidence: float | None = None


class UploadResponse(BaseModel):
    filename: str
    chunks_inserted: int
    entities_extracted: int
    relationships_found: int
    message: str


class EvaluatedQueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    retrieval_type: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    overall_score: float


class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    entities: int
    uploaded_at: str | None


class HealthResponse(BaseModel):
    status: str
    postgres: str
    neo4j: str
    cache_size: int
