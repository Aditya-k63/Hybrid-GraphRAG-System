from app.ingestion.chunker import _merge_tails, _normalize, chunk_text


class TestNormalize:
    def test_collapses_whitespace_keeps_paragraphs(self):
        result = _normalize("hello   world\n\n\n\nsecond  para")
        assert "hello world" in result
        assert "\n\n" in result
        assert "\n\n\n\n" not in result

    def test_strips_edges(self):
        assert _normalize("   padded   ") == "padded"


class TestChunkText:
    def test_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_short_text_is_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_default_bounds(self):
        text = "This is a test sentence with enough words. " * 50
        chunks = chunk_text(text)
        assert len(chunks) > 0
        assert all(len(c) <= 700 for c in chunks)

    def test_explicit_token_budget(self):
        text = "testing adaptive chunking boundaries please split here. " * 40
        chunks = chunk_text(text, chunk_size_tokens=10, overlap_tokens=0)
        assert all(len(c) <= 40 for c in chunks)


class TestMergeTails:
    def test_merges_tiny_tail(self):
        before = ["longer chunk " * 30, "tiny"]
        merged = _merge_tails(before, limit=600)
        assert len(merged) == 1
        assert merged[0].endswith("tiny")

    def test_keeps_tail_when_merge_overflows(self):
        before = ["x" * 597, "tiny"]
        merged = _merge_tails(before, limit=600)
        assert len(merged) == 2