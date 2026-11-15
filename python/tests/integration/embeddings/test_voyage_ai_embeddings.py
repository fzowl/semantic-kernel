# Copyright (c) Microsoft. All rights reserved.

"""Integration tests for Voyage AI embeddings."""

import os

import pytest

from semantic_kernel.connectors.ai.voyage_ai import VoyageAITextEmbedding

pytestmark = pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="Voyage AI API key not set",
)


@pytest.mark.asyncio
async def test_voyage_ai_text_embedding():
    """Test Voyage AI text embeddings with real API."""
    api_key = os.getenv("VOYAGE_API_KEY")
    assert api_key, "VOYAGE_API_KEY environment variable must be set"

    service = VoyageAITextEmbedding(
        ai_model_id="voyage-4-large",
        api_key=api_key,
    )

    texts = [
        "Semantic Kernel is an SDK",
        "Voyage AI provides embeddings",
    ]

    embeddings = await service.generate_embeddings(texts)

    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024  # Default dimension
    print(f"Generated embeddings with dimension: {len(embeddings[0])}")


@pytest.mark.asyncio
async def test_voyage_ai_code_embeddings():
    """Test Voyage AI code embeddings."""
    api_key = os.getenv("VOYAGE_API_KEY")
    assert api_key, "VOYAGE_API_KEY environment variable must be set"

    service = VoyageAITextEmbedding(
        ai_model_id="voyage-code-4",
        api_key=api_key,
    )

    code_snippets = [
        "def hello_world(): print('Hello, World!')",
        "function helloWorld() { console.log('Hello, World!'); }",
    ]

    embeddings = await service.generate_embeddings(code_snippets)

    assert embeddings is not None
    assert len(embeddings) == 2
    print(f"Generated code embeddings with dimension: {len(embeddings[0])}")


@pytest.mark.asyncio
async def test_voyage_ai_multimodal_3_5_embeddings():
    """Test Voyage AI multimodal 3.5 embeddings (supports video input)."""
    from semantic_kernel.connectors.ai.voyage_ai import VoyageAIMultimodalEmbedding

    api_key = os.getenv("VOYAGE_API_KEY")
    assert api_key, "VOYAGE_API_KEY environment variable must be set"

    service = VoyageAIMultimodalEmbedding(
        ai_model_id="voyage-multimodal-3.5",
        api_key=api_key,
    )

    texts = [
        "A photo of a cat sitting on a windowsill",
        "A diagram showing neural network architecture",
    ]

    embeddings = await service.generate_embeddings(texts)

    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024  # Default dimension
    print(f"Generated voyage-multimodal-3.5 embeddings with dimension: {len(embeddings[0])}")


@pytest.mark.asyncio
async def test_voyage_ai_matryoshka_output_dimension():
    """A smaller Matryoshka dimension is honored."""
    from semantic_kernel.connectors.ai.voyage_ai import VoyageAIEmbeddingPromptExecutionSettings

    service = VoyageAITextEmbedding(ai_model_id="voyage-4-large", api_key=os.getenv("VOYAGE_API_KEY"))

    embeddings = await service.generate_embeddings(
        ["Semantic Kernel"], VoyageAIEmbeddingPromptExecutionSettings(output_dimension=256)
    )

    assert embeddings.shape == (1, 256)


@pytest.mark.asyncio
async def test_voyage_ai_contextualized_document_and_query():
    """Documents are auto-chunked, queries are embedded as is."""
    from semantic_kernel.connectors.ai.voyage_ai import (
        VoyageAIContextualizedEmbedding,
        VoyageAIContextualizedEmbeddingPromptExecutionSettings,
    )

    service = VoyageAIContextualizedEmbedding(ai_model_id="voyage-context-4", api_key=os.getenv("VOYAGE_API_KEY"))

    documents = await service.generate_contextualized_chunk_embeddings(["Cats purr.", "Dogs bark."])
    assert [d.index for d in documents] == [0, 1]
    assert all(len(d.chunk_texts) == len(d.embeddings) >= 1 for d in documents)

    queries = await service.generate_contextualized_embeddings(
        ["What do cats do?"], VoyageAIContextualizedEmbeddingPromptExecutionSettings(input_type="query")
    )
    assert queries.shape == (1, 1024)


@pytest.mark.asyncio
async def test_voyage_ai_rerank_with_top_k():
    """rerank-3 returns top_k results, most relevant first."""
    from semantic_kernel.connectors.ai.voyage_ai import VoyageAIReranker, VoyageAIRerankPromptExecutionSettings

    service = VoyageAIReranker(ai_model_id="rerank-3", api_key=os.getenv("VOYAGE_API_KEY"))

    results = await service.rerank(
        "capital of France",
        ["Paris is the capital of France", "Berlin", "Rome"],
        VoyageAIRerankPromptExecutionSettings(top_k=2),
    )

    assert len(results) == 2
    assert results[0].index == 0
    assert results[0].relevance_score >= results[1].relevance_score
