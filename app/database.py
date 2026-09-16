import logging

logger = logging.getLogger(__name__)

_connection_pool = None
_pool_failed = False


def _get_conn_kwargs():
    from app.config import settings
    if not settings.DB_HOST or not settings.DB_PASSWORD:
        return None
    return {
        "dbname": settings.DB_NAME,
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "sslmode": settings.DB_SSLMODE,
    }


def get_pool():
    global _connection_pool, _pool_failed
    if _pool_failed:
        return None
    if _connection_pool is None:
        kwargs = _get_conn_kwargs()
        if kwargs is None:
            logger.warning("DB not configured, running without database")
            _pool_failed = True
            return None
        try:
            from psycopg2 import pool
            _connection_pool = pool.ThreadedConnectionPool(
                minconn=2, maxconn=5, **kwargs,
            )
            logger.info("PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to create DB pool: {e}")
            _pool_failed = True
            return None
    return _connection_pool


def get_connection():
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Database not available")
    from pgvector.psycopg2 import register_vector
    conn = pool.getconn()
    register_vector(conn)
    return conn


def release_connection(conn):
    pool = get_pool()
    if pool and conn:
        try:
            pool.putconn(conn)
        except Exception:
            pass


def db_available():
    return get_pool() is not None


def init_db():
    if not db_available():
        return
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""CREATE TABLE IF NOT EXISTS document_sections (
            id BIGSERIAL PRIMARY KEY, content TEXT NOT NULL, meta JSONB, embedding VECTOR(384));""")
        cur.execute("""CREATE TABLE IF NOT EXISTS search_analytics (
            id BIGSERIAL PRIMARY KEY, question TEXT, retrieval_type VARCHAR(20),
            chunks_used INT, latency_ms FLOAT, created_at TIMESTAMP DEFAULT NOW());""")
        cur.execute("""CREATE TABLE IF NOT EXISTS rag_evaluations (
            id BIGSERIAL PRIMARY KEY, question TEXT, answer TEXT, faithfulness FLOAT,
            answer_relevance FLOAT, context_precision FLOAT, overall_score FLOAT,
            created_at TIMESTAMP DEFAULT NOW());""")
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_embedding ON document_sections USING hnsw (embedding vector_cosine_ops);")
        except Exception:
            conn.rollback()
        conn.commit()
        cur.close()
        logger.info("PostgreSQL tables initialized")
    except Exception as e:
        logger.error(f"DB init failed: {e}")
    finally:
        release_connection(conn)
