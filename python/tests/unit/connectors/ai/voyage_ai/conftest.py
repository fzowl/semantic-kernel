# Copyright (c) Microsoft. All rights reserved.


import pytest


@pytest.fixture(scope="session")
def voyage_ai_unit_test_env(monkeypatch_session):
    """Fixture to set VoyageAI environment variables for testing."""
    monkeypatch_session.setenv("VOYAGE_API_KEY", "test-api-key")
    monkeypatch_session.setenv("VOYAGE_EMBEDDING_MODEL_ID", "voyage-3.5")
    monkeypatch_session.setenv("VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL_ID", "voyage-context-4")
    monkeypatch_session.setenv("VOYAGE_RERANKER_MODEL_ID", "rerank-3")
    monkeypatch_session.setenv("VOYAGE_MULTIMODAL_EMBEDDING_MODEL_ID", "voyage-multimodal-3.5")

    return {
        "VOYAGE_API_KEY": "test-api-key",
        "VOYAGE_EMBEDDING_MODEL_ID": "voyage-3.5",
        "VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL_ID": "voyage-context-4",
        "VOYAGE_RERANKER_MODEL_ID": "rerank-3",
        "VOYAGE_MULTIMODAL_EMBEDDING_MODEL_ID": "voyage-multimodal-3.5",
    }


@pytest.fixture
def voyage_ai_service():
    """Fixture for mock VoyageAI service."""
    pytest.importorskip("voyageai", reason="voyageai package not installed")


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch fixture."""
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(autouse=True)
def fake_voyage_tokenizer(monkeypatch):
    """Count one token per whitespace separated word instead of downloading the real tokenizer."""
    voyageai = pytest.importorskip("voyageai", reason="voyageai package not installed")

    class _Encoding:
        def __init__(self, text: str):
            self.ids = list(range(len(text.split())))

    monkeypatch.setattr(
        voyageai.AsyncClient, "tokenize", lambda self, texts, model=None: [_Encoding(text) for text in texts]
    )
