import json
import logging
import re
import threading
import time

from app.config import settings

logger = logging.getLogger(__name__)

ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "CONCEPT", "EVENT", "TECHNOLOGY"}

ENTITY_PROMPT = """Extract all named entities and the relationships between them from the following text.
For each entity provide:
- name: the entity name
- type: one of PERSON, ORGANIZATION, LOCATION, DATE, CONCEPT, EVENT, TECHNOLOGY
- description: a brief 1-sentence description

For each relationship provide:
- source: name of the source entity
- target: name of the target entity
- relation: a short verb phrase describing the connection

Return ONLY valid JSON matching this schema:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}], "relationships": [{{"source": "...", "target": "...", "relation": "..."}}]}}

Text:
{text}"""


class ExtractionError(Exception):
    """Raised when structured LLM extraction exhausts its retries."""


class ExtractionCircuitBreaker:
    """Opens after N consecutive failures so a flaky LLM provider does not
    silently empty the knowledge graph; resets after a cooldown window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        with self._lock:
            if time.monotonic() < self._open_until:
                return True
            if self._open_until and time.monotonic() >= self._open_until:
                self._open_until = 0.0
                self._consecutive_failures = 0
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= settings.ENTITY_CIRCUIT_BREAKER_THRESHOLD:
                self._open_until = time.monotonic() + settings.ENTITY_CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    "Entity extraction circuit breaker opened for %ss after %d consecutive failures",
                    settings.ENTITY_CIRCUIT_BREAKER_COOLDOWN,
                    self._consecutive_failures,
                )


_breaker = ExtractionCircuitBreaker()


def _extract_json_object(content: str) -> dict:
    """Lenient JSON extraction: strips code fences and finds the first JSON object."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model output is not a JSON object")
    return parsed


def _call_json_mode(client, messages: list[dict], max_tokens: int) -> dict:
    from groq import Groq

    if not isinstance(client, Groq):
        raise TypeError("expected a Groq client")
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return _extract_json_object(response.choices[0].message.content)


def _extract_entities_llm_raw(text: str) -> dict:
    """Call the LLM in JSON mode with bounded retry/backoff for rate limits."""
    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)
    truncated = text[: settings.ENTITY_MAX_TEXT_CHARS]
    last_error: Exception | None = None

    for attempt in range(settings.ENTITY_MAX_RETRIES):
        try:
            return _call_json_mode(
                client,
                [{"role": "user", "content": ENTITY_PROMPT.format(text=truncated)}],
                max_tokens=2000,
            )
        except Exception as e:
            last_error = e
            status = getattr(e, "status_code", None)
            is_rate_limit = status in (429, 529) or "429" in str(e)
            if is_rate_limit and attempt < settings.ENTITY_MAX_RETRIES - 1:
                wait = settings.ENTITY_RETRY_BACKOFF ** attempt
                logger.warning("LLM extraction rate limited; retrying in %ss", wait)
                time.sleep(wait)
                continue

    raise ExtractionError(f"structured extraction failed after {settings.ENTITY_MAX_RETRIES} attempts: {last_error}")


def _clean_entities(raw: list) -> list[dict]:
    """Deduplicate, type-normalize, and drop low-quality entity rows."""
    seen: set[str] = set()
    cleaned: list[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        entity_type = str(row.get("type", "") or "CONCEPT").strip().upper() or "CONCEPT"
        if entity_type not in ENTITY_TYPES:
            entity_type = "CONCEPT"
        description = str(row.get("description", "")).strip()
        if settings.ENTITY_REQUIRE_DESCRIPTION and not description:
            continue
        if not description:
            description = f"Extracted entity: {name}"
        seen.add(key)
        cleaned.append({"name": name, "type": entity_type, "description": description})
    return cleaned


def _clean_relationships(raw: list, entity_names: set[str]) -> list[dict]:
    """Keep only relationships whose endpoints exist among the cleaned entities."""
    cleaned: list[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        if not source or not target:
            continue
        if source.lower() not in entity_names or target.lower() not in entity_names:
            continue
        relation = str(row.get("relation", "")).strip() or "related"
        cleaned.append({"source": source, "target": target, "relation": relation})
    return cleaned


def _merge_entities(*groups: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for group in groups:
        for entity in group:
            key = entity["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(entity)
    return merged


class SpaCyNER:
    """Lightweight NER using spaCy for entity pre-extraction."""

    def __init__(self):
        self.nlp = None
        self._load_model()

    def _load_model(self):
        try:
            import spacy
            self.nlp = spacy.load(settings.SPACY_MODEL)
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Install with: python -m spacy download %s",
                settings.SPACY_MODEL,
                settings.SPACY_MODEL,
            )
            self.nlp = None
        except Exception as e:
            logger.warning("Failed to load spaCy: %s", e)
            self.nlp = None

    def extract_entities(self, text: str) -> list[dict]:
        if self.nlp is None:
            return []
        doc = self.nlp(text[:5000])
        entities = []
        for ent in doc.ents:
            entity_type = self._map_spacy_type(ent.label_)
            entities.append({
                "name": ent.text,
                "type": entity_type,
                "description": f"Extracted entity: {ent.text}",
            })
        return entities

    def _map_spacy_type(self, label: str) -> str:
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "EVENT": "EVENT",
            "WORK_OF_ART": "TECHNOLOGY",
            "PRODUCT": "TECHNOLOGY",
            "FAC": "LOCATION",
        }
        return mapping.get(label, "CONCEPT")


_spacy_ner = None


def get_spacy_ner() -> SpaCyNER | None:
    global _spacy_ner
    if _spacy_ner is None:
        _spacy_ner = SpaCyNER()
    return _spacy_ner


def extract_entities_with_spacy(text: str) -> dict:
    """Extract entities using spaCy only."""
    spacy_ner = get_spacy_ner()
    spacy_entities: list[dict] = []
    if spacy_ner and settings.USE_SPACY:
        spacy_entities = spacy_ner.extract_entities(text)
        logger.info("spaCy extracted %s entities", len(spacy_entities))
    return {"entities": spacy_entities, "spacy_count": len(spacy_entities)}


def extract_entities_llm(text: str) -> dict:
    """Structured LLM entity + relationship extraction with validation and a circuit breaker."""
    if not settings.ENABLE_LIVE_LLM:
        return {"entities": [], "relationships": []}
    if _breaker.is_open():
        logger.warning("Entity extraction skipped while circuit breaker is open")
        return {"entities": [], "relationships": []}

    try:
        raw = _extract_entities_llm_raw(text)
    except ExtractionError:
        _breaker.record_failure()
        logger.error("LLM entity extraction failed, falling back to spaCy only")
        return {"entities": [], "relationships": []}

    _breaker.record_success()
    entities = _clean_entities(raw.get("entities", []))
    entity_names = {entity["name"].lower() for entity in entities}
    relationships = _clean_relationships(raw.get("relationships", []), entity_names)
    logger.info("LLM extraction produced %s entities, %s relationships", len(entities), len(relationships))
    return {"entities": entities, "relationships": relationships}


def extract_entities(text: str) -> dict:
    """Combined entity extraction: spaCy for NER + structured LLM for entities and relationships."""
    spacy_data = extract_entities_with_spacy(text)
    llm_data = extract_entities_llm(text)

    entities = _merge_entities(spacy_data["entities"], llm_data["entities"])
    entity_names = {entity["name"].lower() for entity in entities}
    relationships = _clean_relationships(llm_data["relationships"], entity_names)

    logger.info(
        "Combined extraction: %s entities, %s relationships",
        len(entities),
        len(relationships),
    )
    return {"entities": entities, "relationships": relationships}


def extract_query_entities(query: str) -> list[str]:
    """Extract key entities from a query using spaCy, with an LLM fallback."""
    spacy_ner = get_spacy_ner()
    if spacy_ner and settings.USE_SPACY:
        doc = spacy_ner.nlp(query)
        entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "LOC")]
        if entities:
            return entities

    if not settings.ENABLE_LIVE_LLM:
        return []

    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"Extract key named entities from: {query}. Return ONLY a JSON array of strings.",
            }],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        data = _extract_json_object(content)
        entities = data.get("entities", []) if isinstance(data, dict) else data
        if isinstance(entities, list):
            return [str(item).strip() for item in entities if str(item).strip()]
        return []
    except Exception as e:
        logger.error("LLM query entity extraction failed: %s", e)
        return []