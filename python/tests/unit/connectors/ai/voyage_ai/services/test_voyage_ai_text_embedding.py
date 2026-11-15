# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import AsyncMock, MagicMock

import pytest
from numpy import ndarray

from semantic_kernel.connectors.ai.voyage_ai import (
    VoyageAIEmbeddingPromptExecutionSettings,
    VoyageAITextEmbedding,
)
from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError, ServiceResponseException


def test_init(voyage_ai_unit_test_env):
    """Test VoyageAI text embedding initialization."""
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
    )

    assert embedding_service.ai_model_id == "voyage-3.5"
    assert embedding_service.service_id == "voyage-3.5"
    assert embedding_service.aclient is not None


def test_init_with_service_id(voyage_ai_unit_test_env):
    """Test initialization with custom service ID."""
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        service_id="custom-service",
        api_key="test-api-key",
    )

    assert embedding_service.ai_model_id == "voyage-3.5"
    assert embedding_service.service_id == "custom-service"


def test_init_from_env(voyage_ai_unit_test_env):
    """Test initialization from environment variables."""
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
    )

    assert embedding_service.ai_model_id == "voyage-3.5"


def test_init_uses_embedding_model_id_from_settings(voyage_ai_unit_test_env):
    """Test that ai_model_id falls back to the VOYAGE_EMBEDDING_MODEL_ID setting."""
    embedding_service = VoyageAITextEmbedding(
        api_key="test-api-key",
    )

    assert embedding_service.ai_model_id == "voyage-3.5"


def test_init_without_model_id_raises(monkeypatch):
    """Test that omitting the model everywhere raises a clear initialization error."""
    from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError

    monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
    monkeypatch.delenv("VOYAGE_EMBEDDING_MODEL_ID", raising=False)

    with pytest.raises(ServiceInitializationError):
        VoyageAITextEmbedding(api_key="test-api-key", env_file_path="non_existent.env")


def test_init_with_custom_endpoint(voyage_ai_unit_test_env):
    """Test initialization with custom endpoint."""
    custom_endpoint = "https://custom-endpoint.com/v1"
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
        endpoint=custom_endpoint,
    )

    assert embedding_service.endpoint == custom_endpoint


def test_prompt_execution_settings_class():
    """Test getting prompt execution settings class."""
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
    )

    assert embedding_service.get_prompt_execution_settings_class() == VoyageAIEmbeddingPromptExecutionSettings


@pytest.mark.asyncio
async def test_generate_embeddings(voyage_ai_unit_test_env):
    """Test generating embeddings."""
    # Mock response
    mock_response = MagicMock()
    mock_response.embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
    )

    # Mock the aclient
    embedding_service._aclient.embed = AsyncMock(return_value=mock_response)

    texts = ["hello world", "goodbye world"]
    embeddings = await embedding_service.generate_embeddings(texts)

    assert isinstance(embeddings, ndarray)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    embedding_service._aclient.embed.assert_called_once()


@pytest.mark.asyncio
async def test_generate_embeddings_with_settings(voyage_ai_unit_test_env):
    """Test generating embeddings with execution settings."""
    # Mock response
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3]]

    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
    )

    # Mock the aclient
    embedding_service._aclient.embed = AsyncMock(return_value=mock_response)

    settings = VoyageAIEmbeddingPromptExecutionSettings(
        input_type="query",
        truncation=True,
        output_dimension=1024,
    )

    texts = ["test text"]
    embeddings = await embedding_service.generate_embeddings(texts, settings=settings)

    assert isinstance(embeddings, ndarray)
    embedding_service._aclient.embed.assert_called_once()


@pytest.mark.asyncio
async def test_generate_embeddings_with_different_output_dtype(voyage_ai_unit_test_env):
    """Test generating embeddings with different output dtype."""
    # Mock response
    mock_response = MagicMock()
    mock_response.embeddings = [[1, 2, 3]]  # int8 format

    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
    )

    # Mock the aclient
    embedding_service._aclient.embed = AsyncMock(return_value=mock_response)

    settings = VoyageAIEmbeddingPromptExecutionSettings(
        output_dtype="int8",
    )

    texts = ["test text"]
    embeddings = await embedding_service.generate_embeddings(texts, settings=settings)

    assert isinstance(embeddings, ndarray)
    embedding_service._aclient.embed.assert_called_once()


@pytest.mark.asyncio
async def test_generate_embeddings_failure(voyage_ai_unit_test_env):
    """Test handling of embedding generation failure."""
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
    )

    # Mock the aclient to raise an exception
    embedding_service._aclient.embed = AsyncMock(side_effect=Exception("API error"))

    texts = ["test text"]

    with pytest.raises(ServiceResponseException):
        await embedding_service.generate_embeddings(texts)


def test_service_url():
    """Test service URL retrieval."""
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
    )

    # No explicit endpoint: the voyageai package picks the default host from the API key.
    assert embedding_service.service_url() is None


def test_service_url_with_custom_endpoint():
    """Test service URL with custom endpoint."""
    custom_endpoint = "https://custom.api.com/v1"
    embedding_service = VoyageAITextEmbedding(
        ai_model_id="voyage-3.5",
        api_key="test-api-key",
        endpoint=custom_endpoint,
    )

    assert embedding_service.service_url() == custom_endpoint


def test_unknown_model_is_rejected():
    with pytest.raises(ServiceInitializationError, match="Unknown Voyage AI text embedding model"):
        VoyageAITextEmbedding(ai_model_id="voyage-nonexistent", api_key="test-api-key")


def _batching_service(model: str, embed_side_effect) -> VoyageAITextEmbedding:
    service = VoyageAITextEmbedding(ai_model_id=model, api_key="test-api-key")
    service._aclient.embed = AsyncMock(side_effect=embed_side_effect)
    return service


def _embed_each_text_as_its_length(texts, **kwargs):
    response = MagicMock()
    response.embeddings = [[float(len(text))] for text in texts]
    return response


@pytest.mark.asyncio
async def test_generate_embeddings_splits_by_token_limit_and_keeps_order(voyage_ai_unit_test_env, monkeypatch):
    from semantic_kernel.connectors.ai.voyage_ai.services import voyage_ai_text_embedding

    monkeypatch.setitem(voyage_ai_text_embedding.MAX_TOKENS_PER_REQUEST, "voyage-3.5", 5)
    service = _batching_service("voyage-3.5", _embed_each_text_as_its_length)
    # The fixture counts one token per word.
    texts = ["a b c", "d e", "f g h i", "j", "k l"]

    embeddings = await service.generate_embeddings(texts)

    sent = [call.kwargs["texts"] for call in service._aclient.embed.call_args_list]
    assert sent == [["a b c", "d e"], ["f g h i", "j"], ["k l"]]
    assert [row[0] for row in embeddings] == [float(len(t)) for t in texts]


@pytest.mark.asyncio
async def test_generate_embeddings_splits_at_1000_inputs(voyage_ai_unit_test_env):
    service = _batching_service("voyage-4-lite", _embed_each_text_as_its_length)
    texts = [str(i) for i in range(2001)]

    embeddings = await service.generate_embeddings(texts)

    assert [len(call.kwargs["texts"]) for call in service._aclient.embed.call_args_list] == [1000, 1000, 1]
    assert len(embeddings) == 2001
    assert [row[0] for row in embeddings] == [float(len(t)) for t in texts]


@pytest.mark.asyncio
async def test_model_without_documented_token_cap_only_splits_by_input_count(voyage_ai_unit_test_env):
    service = _batching_service("voyage-code-4", _embed_each_text_as_its_length)
    service._aclient.tokenize = MagicMock(side_effect=AssertionError("tokens must not be counted"))

    await service.generate_embeddings(["x"] * 1001)

    assert [len(call.kwargs["texts"]) for call in service._aclient.embed.call_args_list] == [1000, 1]


@pytest.mark.asyncio
async def test_failure_in_a_later_batch_is_not_swallowed(voyage_ai_unit_test_env):
    responses = [_embed_each_text_as_its_length(["x"] * 1000), Exception("rate limited")]
    service = _batching_service("voyage-4-lite", responses)

    with pytest.raises(ServiceResponseException, match="rate limited"):
        await service.generate_embeddings(["x"] * 1001)
