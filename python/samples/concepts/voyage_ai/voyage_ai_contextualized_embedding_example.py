# Copyright (c) Microsoft. All rights reserved.

"""Example demonstrating Voyage AI by MongoDB contextualized embeddings with Semantic Kernel."""

import asyncio
import os

from semantic_kernel.connectors.ai.voyage_ai import (
    VoyageAIContextualizedEmbedding,
    VoyageAIContextualizedEmbeddingPromptExecutionSettings,
)


async def main():
    """Run the Voyage AI by MongoDB contextualized embedding example."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("Please set the VOYAGE_API_KEY environment variable")

    service = VoyageAIContextualizedEmbedding(ai_model_id="voyage-context-4", api_key=api_key)

    # Document mode (default): every string is one full document. The API chunks each document and
    # returns one embedding per chunk, together with the chunk texts.
    documents = [
        "Semantic Kernel is an open-source SDK for building agents. It supports plugins and planners.",
        "Voyage AI by MongoDB provides embedding and reranking models.",
    ]
    results = await service.generate_contextualized_chunk_embeddings(documents)
    for result in results:
        print(f"Document {result.index}: {len(result.chunk_texts)} chunk(s), embeddings {result.embeddings.shape}")

    # Query mode: every string is one query, there is no chunking.
    query_settings = VoyageAIContextualizedEmbeddingPromptExecutionSettings(input_type="query")
    query_embeddings = await service.generate_contextualized_embeddings(
        ["What does Semantic Kernel support?"], query_settings
    )
    print(f"Query embeddings: {query_embeddings.shape}")


if __name__ == "__main__":
    asyncio.run(main())
