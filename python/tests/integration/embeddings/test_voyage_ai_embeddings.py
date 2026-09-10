# Copyright (c) Microsoft. All rights reserved.

"""Integration tests for VoyageAI embeddings."""

import os

import pytest

from semantic_kernel.connectors.ai.voyage_ai import VoyageAITextEmbedding

pytestmark = pytest.mark.skipif(
    not os.getenv("VOYAGE_AI_API_KEY"),
    reason="VoyageAI API key not set",
)


@pytest.mark.asyncio
async def test_voyage_ai_text_embedding():
    """Test VoyageAI text embeddings with real API."""
    api_key = os.getenv("VOYAGE_AI_API_KEY")
    assert api_key, "VOYAGE_AI_API_KEY environment variable must be set"

    service = VoyageAITextEmbedding(
        ai_model_id="voyage-3-large",
        api_key=api_key,
    )

    texts = [
        "Semantic Kernel is an SDK",
        "VoyageAI provides embeddings",
    ]

    embeddings = await service.generate_embeddings(texts)

    assert embeddings is not None
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0  # Should have embedding dimensions
    print(f"Generated embeddings with dimension: {len(embeddings[0])}")


@pytest.mark.asyncio
async def test_voyage_ai_code_embeddings():
    """Test VoyageAI code embeddings."""
    api_key = os.getenv("VOYAGE_AI_API_KEY")
    assert api_key, "VOYAGE_AI_API_KEY environment variable must be set"

    service = VoyageAITextEmbedding(
        ai_model_id="voyage-code-3",
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
    """Test VoyageAI multimodal 3.5 embeddings (preview model with video support)."""
    from semantic_kernel.connectors.ai.voyage_ai import VoyageAIMultimodalEmbedding

    api_key = os.getenv("VOYAGE_AI_API_KEY")
    assert api_key, "VOYAGE_AI_API_KEY environment variable must be set"

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
