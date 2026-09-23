import json
import re


def extract_json_object(content: str) -> dict:
    """Leniently extract the first JSON object from a model response,
    tolerating prose and markdown code fences. Raises ValueError on failure."""
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