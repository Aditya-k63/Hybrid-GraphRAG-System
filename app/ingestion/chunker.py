import logging
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.text import estimate_tokens

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_TAIL_MERGE_RATIO = 0.12
_TAIL_MIN_CHARS = 40


def _normalize(text: str) -> str:
    """Collapse stray whitespace produces by PDF extraction while keeping paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _merge_tails(chunks: list[str], limit: int) -> list[str]:
    """Merge short trailing chunks into their predecessor so no tiny orphan fragments
    pollute the vector store (toolSearch.ts-style quality guard)."""
    if len(chunks) < 2:
        return chunks
    merged = chunks[:]
    while len(merged) > 1:
        tail = merged[-1]
        prev = merged[-2]
        threshold = max(_TAIL_MIN_CHARS, int(limit * _TAIL_MERGE_RATIO))
        if len(tail) < threshold and len(prev) + len(tail) + 1 <= limit:
            merged[-2] = prev + " " + tail
            merged.pop()
            continue
        break
    return merged


def chunk_text(text: str, chunk_size_tokens: int | None = None, overlap_tokens: int | None = None) -> list[str]:
    """Token-aware recursive chunker that splits on semantic boundaries.

    Sizing is derived from a rough token estimate (chars_per_token=4, the
    toolSearch.ts CHARS_PER_TOKEN pattern). Defaults honour settings.CHUNK_SIZE
    / CHUNK_OVERLAP (characters) so existing callers keep identical bounds.
    """
    text = _normalize(text)
    if not text:
        return []

    size_tokens = chunk_size_tokens or max(1, settings.CHUNK_SIZE // 4)
    overlap_tokens = overlap_tokens if overlap_tokens is not None else max(1, settings.CHUNK_OVERLAP // 4)

    chunk_size = size_tokens * 4
    chunk_overlap = overlap_tokens * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_SEPARATORS,
    )
    chunks = [piece.strip() for piece in splitter.split_text(text) if piece.strip()]
    chunks = _merge_tails(chunks, chunk_size)

    logger.info(
        "Split text into %s chunks (target %s tokens, overlap %s tokens, ~%s total tokens)",
        len(chunks),
        size_tokens,
        overlap_tokens,
        estimate_tokens(text),
    )
    return chunks