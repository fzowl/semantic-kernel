# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import AsyncMock, MagicMock

import pytest
from numpy import ndarray

from semantic_kernel.connectors.ai.voyage_ai import (
    VoyageAIContextualizedEmbedding,
    VoyageAIContextualizedEmbeddingPromptExecutionSettings,
)
from semantic_kernel.exceptions.service_exceptions import (
    ServiceInitializationError,
    ServiceInvalidRequestError,
    ServiceResponseException,
)


def _service(*results) -> VoyageAIContextualizedEmbedding:
    service = VoyageAIContextualizedEmbedding(ai_model_id="voyage-context-4", api_key="test-api-key")
    response = MagicMock()
    response.results = list(results)
    service._aclient.contextualized_embed = AsyncMock(return_value=response)
    return service


def _result(chunk_texts, embeddings):
    result = MagicMock()
    result.chunk_texts = chunk_texts
    result.embeddings = embeddings
    return result


def test_init_without_model_id_raises(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
    monkeypatch.delenv("VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL_ID", raising=False)

    with pytest.raises(ServiceInitializationError):
        VoyageAIContextualizedEmbedding(env_file_path="non_existent.env")


@pytest.mark.asyncio
async def test_document_mode_sends_flat_documents_with_auto_chunking(voyage_ai_unit_test_env):
    service = _service(_result(["a", "b"], [[0.1], [0.2]]), _result(["c"], [[0.3]]))

    embeddings = await service.generate_contextualized_embeddings(["doc one", "doc two"])

    kwargs = service._aclient.contextualized_embed.call_args.kwargs
    assert kwargs["inputs"] == ["doc one", "doc two"]
    assert kwargs["input_type"] == "document"
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == 32_000
    assert isinstance(embeddings, ndarray)
    assert embeddings.shape == (3, 1)


@pytest.mark.asyncio
async def test_generate_embeddings_treats_each_text_as_a_document(voyage_ai_unit_test_env):
    service = _service(_result(["a"], [[0.1]]), _result(["b"], [[0.2]]))

    embeddings = await service.generate_embeddings(["a", "b"])

    assert service._aclient.contextualized_embed.call_args.kwargs["inputs"] == ["a", "b"]
    assert embeddings.shape == (2, 1)


@pytest.mark.asyncio
async def test_chunk_embeddings_returns_chunk_texts_per_document(voyage_ai_unit_test_env):
    service = _service(_result(["a", "b"], [[0.1], [0.2]]), _result(["c"], [[0.3]]))

    results = await service.generate_contextualized_chunk_embeddings(["doc one", "doc two"])

    assert [r.index for r in results] == [0, 1]
    assert results[0].chunk_texts == ["a", "b"]
    assert results[0].embeddings.shape == (2, 1)
    assert results[1].chunk_texts == ["c"]


@pytest.mark.asyncio
async def test_query_mode_wraps_each_query_and_skips_auto_chunking(voyage_ai_unit_test_env):
    service = _service(_result(None, [[0.1]]), _result(None, [[0.2]]))
    settings = VoyageAIContextualizedEmbeddingPromptExecutionSettings(input_type="query")

    embeddings = await service.generate_contextualized_embeddings(["q1", "q2"], settings)

    kwargs = service._aclient.contextualized_embed.call_args.kwargs
    assert kwargs["inputs"] == [["q1"], ["q2"]]
    assert kwargs["input_type"] == "query"
    assert "enable_auto_chunking" not in kwargs
    assert "chunk_size" not in kwargs
    assert embeddings.shape == (2, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["document", "query"])
async def test_pre_chunked_input_is_rejected_before_the_request(voyage_ai_unit_test_env, input_type):
    service = _service()
    settings = VoyageAIContextualizedEmbeddingPromptExecutionSettings(input_type=input_type)

    with pytest.raises(ServiceInvalidRequestError, match="flat list of strings"):
        await service.generate_contextualized_embeddings([["chunk1", "chunk2"]], settings)  # type: ignore[list-item]

    service._aclient.contextualized_embed.assert_not_called()


@pytest.mark.asyncio
async def test_empty_input_makes_no_request(voyage_ai_unit_test_env):
    service = _service()

    embeddings = await service.generate_contextualized_embeddings([])

    assert len(embeddings) == 0
    service._aclient.contextualized_embed.assert_not_called()


@pytest.mark.asyncio
async def test_api_failure_is_wrapped(voyage_ai_unit_test_env):
    service = _service()
    service._aclient.contextualized_embed = AsyncMock(side_effect=Exception("API error"))

    with pytest.raises(ServiceResponseException, match="API error"):
        await service.generate_contextualized_embeddings(["doc"])
