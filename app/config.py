import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    DB_NAME: str = "graphrag"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # Groq
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.1-8b-instant"

    # Auth
    API_KEY: str = "change-me-in-production"
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60

    # Embedding
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Chunking
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 50

    # Retrieval
    VECTOR_TOP_K: int = 10
    BM25_TOP_K: int = 10
    GRAPH_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    RRF_K: int = 60
    MAX_HOPS: int = 3

    # spaCy
    USE_SPACY: bool = True
    SPACY_MODEL: str = "en_core_web_sm"

    # Cache
    CACHE_MAX_SIZE: int = 200

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: int = 60

    # Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    MAX_CHUNKS: int = 500

    # Latency tracking
    ENABLE_LATENCY_TRACKING: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()