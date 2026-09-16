import logging

logger = logging.getLogger(__name__)


def graph_retrieve(query: str, max_hops: int = 3) -> list[dict]:
    """Retrieve from Neo4j knowledge graph with multi-hop traversal."""
    try:
        from app import graph as graph_module
        from app.ingestion.entity_extractor import extract_query_entities
        from app.retrieval.tracker import tracker

        graph_search_fn = getattr(graph_module, "graph_search", None)
        get_entity_context_fn = getattr(graph_module, "get_entity_context", None)
        if not callable(graph_search_fn) or not callable(get_entity_context_fn):
            logger.warning("Graph helpers unavailable; skipping graph retrieval")
            return []

        entities = extract_query_entities(query)
        if not entities:
            logger.info("No entities found in query for graph search")
            return []

        logger.info(f"Query entities for graph (max_hops={max_hops}): {entities}")

        graph_results = graph_search_fn(entities, max_hops=max_hops)
        context_texts = get_entity_context_fn(entities)

        tracker.record("graph", 1)

        seen = set()
        results = []
        for item in graph_results or []:
            name = item.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            results.append({
                "content": f"{name} ({item.get('type', 'Entity')}): {item.get('description', '')}",
                "score": 1.0 / (1 + item.get("distance", 0)),
                "source": "graph",
                "entity": name,
                "hops": item.get("distance", 0),
            })

        for ctx in context_texts or []:
            first_line = ctx.split("|")[0].strip()
            if first_line not in seen:
                seen.add(first_line)
                results.append({"content": ctx, "score": 0.8, "source": "graph_context"})

        logger.info(f"Graph retrieval returned {len(results)} results for {len(entities)} entities with {max_hops} hops")
        return results[:20]
    except Exception as e:
        logger.warning(f"Graph retrieval failed (falling back to vector): {e}")
        return []


def multi_hop_query(query: str, hop_chain: list[str]) -> list[dict]:
    """Execute a multi-hop query following a chain of entity types."""
    try:
        from app.graph import get_driver
        driver = get_driver()
        if not callable(getattr(driver, "session", None)):
            return []
        results = []
        with driver.session() as session:
            for i, entity_type in enumerate(hop_chain):
                cypher = f"""
                    MATCH (e:Entity {{type: $type}})
                    WHERE EXISTS {{ MATCH (e)-[:RELATED_TO]-() }}
                    RETURN e.name AS name, e.type AS type, e.description AS description
                    LIMIT 10
                """
                record_result = session.run(cypher, type=entity_type)
                for record in record_result:
                    results.append({
                        "name": record["name"],
                        "type": record["type"],
                        "description": record["description"],
                        "hop": i,
                        "source": "multi_hop",
                    })
        return results
    except Exception as e:
        logger.error(f"Multi-hop query failed: {e}")
        return []
