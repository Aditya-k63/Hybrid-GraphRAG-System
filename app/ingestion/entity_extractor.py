import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class SpaCyNER:
    """Lightweight NER using spaCy for entity pre-extraction."""

    def __init__(self):
        self.nlp = None
        self._load_model()

    def _load_model(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        except Exception as e:
            logger.warning(f"Failed to load spaCy: {e}")
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
    """Extract entities using spaCy, supplementing with Groq LLM."""
    spacy_ner = get_spacy_ner()
    spacy_entities = []
    if spacy_ner:
        spacy_entities = spacy_ner.extract_entities(text)
        logger.info(f"spaCy extracted {len(spacy_entities)} entities")

    return {
        "entities": spacy_entities,
        "spacy_count": len(spacy_entities),
    }


def extract_entities_llm(text: str) -> dict:
    """Extract entities using Groq LLM."""
    from app.config import settings
    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)
    ENTITY_PROMPT = """Extract all named entities from the following text. For each entity, provide:
- name: the entity name
- type: one of PERSON, ORGANIZATION, LOCATION, DATE, CONCEPT, EVENT, TECHNOLOGY
- description: a brief 1-sentence description

Return ONLY valid JSON:
{"entities": [{"name": "...", "type": "...", "description": "..."}], "relationships": []}

Text:
{text}"""

    truncated = text[:3000] if len(text) > 3000 else text
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": ENTITY_PROMPT.format(text=truncated)}],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return {"entities": result.get("entities", []), "relationships": result.get("relationships", [])}
    except Exception as e:
        logger.error(f"LLM entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


def extract_entities(text: str) -> dict:
    """Combined entity extraction: spaCy for NER + Groq for relationships."""
    spacy_data = extract_entities_with_spacy(text)
    llm_data = extract_entities_llm(text)

    # Merge: prefer spaCy entities, add LLM relationships
    entities = spacy_data["entities"]
    if not entities and llm_data["entities"]:
        entities = llm_data["entities"]

    relationships = llm_data["relationships"]

    logger.info(f"Combined extraction: {len(entities)} entities, {len(relationships)} relationships")
    return {"entities": entities, "relationships": relationships}


def extract_query_entities(query: str) -> list[str]:
    """Extract key entities from a query using spaCy + LLM fallback."""
    spacy_ner = get_spacy_ner()
    if spacy_ner:
        doc = spacy_ner.nlp(query)
        entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "LOC")]
        if entities:
            return entities

    from app.config import settings
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": f"Extract key named entities from: {query}. Return ONLY a JSON array."}],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
        return json.loads(content) if isinstance(content, str) else []
    except Exception:
        return []