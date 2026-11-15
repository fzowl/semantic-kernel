# Copyright (c) Microsoft. All rights reserved.

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from numpy import array, ndarray

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.connectors.ai.voyage_ai.services.voyage_ai_base import VoyageAIBase
from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_prompt_execution_settings import (
    VoyageAIContextualizedEmbeddingPromptExecutionSettings,
)
from semantic_kernel.exceptions.service_exceptions import (
    ServiceInitializationError,
    ServiceInvalidRequestError,
    ServiceResponseException,
)

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings


AUTO_CHUNK_SIZE = 32_000


@dataclass
class VoyageAIContextualizedChunkEmbeddings:
    """The contextualized embeddings of one input.

    Attributes:
        index: Position of the input in the request.
        chunk_texts: The chunks the API split the input into (the query itself in ``query`` mode).
        embeddings: One embedding per chunk, same order as ``chunk_texts``.
    """

    index: int
    chunk_texts: list[str]
    embeddings: ndarray


class VoyageAIContextualizedEmbedding(VoyageAIBase, EmbeddingGeneratorBase):
    """Voyage AI by MongoDB Contextualized Embedding Service.

    Generates embeddings that capture both local chunk details and
    global document-level metadata using the voyage-context-4 or voyage-context-3 model.

    Note: Contextualized embeddings do not support truncation. Inputs that
    exceed the model's context length will be rejected.
    """

    def __init__(
        self,
        ai_model_id: str | None = None,
        service_id: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        env_file_path: str | None = None,
        endpoint: str | None = None,
    ):
        """Initialize Voyage AI contextualized embedding service.

        Args:
            ai_model_id: The Voyage AI model ID. Falls back to the ``contextualized_embedding_model_id``
                setting (env var ``VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL_ID``) when not provided.
            service_id: The service ID (optional).
            api_key: The Voyage AI API key (optional, defaults to the ``VOYAGE_API_KEY`` env var).
            client: A pre-configured ``voyageai.AsyncClient`` (optional).
            env_file_path: Path to .env file (optional).
            endpoint: Explicit API base URL (optional, overrides the default derived from the API key).
        """
        # Use contextualized model from settings if not provided
        if not ai_model_id:
            from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_settings import VoyageAISettings

            settings = VoyageAISettings(env_file_path=env_file_path)  # type: ignore[call-arg]
            ai_model_id = settings.contextualized_embedding_model_id

        if not ai_model_id:
            raise ServiceInitializationError(
                "No model ID provided. Set ai_model_id parameter or "
                "VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL_ID environment variable."
            )

        super().__init__(
            ai_model_id=ai_model_id,
            service_id=service_id,
            api_key=api_key,
            client=client,
            env_file_path=env_file_path,
            endpoint=endpoint,
        )

    async def generate_contextualized_embeddings(
        self,
        inputs: list[str],
        settings: "PromptExecutionSettings | None" = None,
        **kwargs: Any,
    ) -> ndarray:
        """Generate contextualized embeddings for the given inputs.

        In ``document`` mode (the default) every string is one full document. The API chunks each document
        (``enable_auto_chunking`` with ``chunk_size`` 32,000) and returns one embedding per chunk, so the
        result has one row per chunk, documents in input order. Use
        ``generate_contextualized_chunk_embeddings`` to get the chunk texts together with the embeddings.

        In ``query`` mode every string is one query and the result has one row per query.

        Args:
            inputs: A flat list of strings, one document (``document`` mode) or one query (``query`` mode)
                per element. Pre-chunked ``list[list[str]]`` input is not accepted.
            settings: Prompt execution settings (optional). ``input_type`` is ``document`` by default.
            kwargs: Additional arguments.

        Returns:
            ndarray: Array of contextualized embeddings.
        """
        results = await self._embed(inputs, settings)
        return array([embedding for result in results for embedding in result.embeddings])

    async def generate_contextualized_chunk_embeddings(
        self,
        inputs: list[str],
        settings: "PromptExecutionSettings | None" = None,
        **kwargs: Any,
    ) -> list[VoyageAIContextualizedChunkEmbeddings]:
        """Generate contextualized embeddings and return them per input, together with the chunk texts.

        Args:
            inputs: A flat list of strings, one document (``document`` mode) or one query (``query`` mode)
                per element.
            settings: Prompt execution settings (optional). ``input_type`` is ``document`` by default.
            kwargs: Additional arguments.

        Returns:
            One result per input, in input order. ``chunk_texts`` holds the chunks the API produced for the
            document, in ``query`` mode it holds the query.
        """
        results = await self._embed(inputs, settings)
        return [
            VoyageAIContextualizedChunkEmbeddings(
                index=index,
                chunk_texts=list(result.chunk_texts) if result.chunk_texts is not None else [],
                embeddings=array(result.embeddings),
            )
            for index, result in enumerate(results)
        ]

    async def _embed(self, inputs: list[str], settings: "PromptExecutionSettings | None") -> list[Any]:
        if not settings:
            settings = VoyageAIContextualizedEmbeddingPromptExecutionSettings()
        else:
            settings = self.get_prompt_execution_settings_from_settings(settings)

        if not inputs:
            return []
        if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
            raise ServiceInvalidRequestError(
                "Contextualized embeddings expect a flat list of strings, one document or one query per "
                "element. Pre-chunked list[list[str]] input is not supported: the API rejects it with "
                "auto-chunking."
            )

        request_settings = settings.prepare_settings_dict()
        input_type = request_settings["input_type"]
        try:
            if input_type == "document":
                response = await self.aclient.contextualized_embed(
                    inputs=inputs,
                    model=self.ai_model_id,
                    enable_auto_chunking=True,
                    chunk_size=AUTO_CHUNK_SIZE,
                    **request_settings,
                )
            else:
                # Auto-chunking only works with input_type "document". A query is a document of one chunk.
                response = await self.aclient.contextualized_embed(
                    inputs=[[query] for query in inputs],
                    model=self.ai_model_id,
                    **request_settings,
                )
            return list(response.results)
        except Exception as e:
            raise ServiceResponseException(f"Voyage AI contextualized embedding request failed: {e}") from e

    @override
    async def generate_embeddings(
        self,
        texts: list[str],
        settings: "PromptExecutionSettings | None" = None,
        **kwargs: Any,
    ) -> ndarray:
        """Generate embeddings for texts, each text being one document (or one query in ``query`` mode).

        Args:
            texts: List of texts to generate embeddings for.
            settings: Prompt execution settings (optional).
            kwargs: Additional arguments.

        Returns:
            ndarray: Array of embeddings.
        """
        return await self.generate_contextualized_embeddings(texts, settings, **kwargs)

    @override
    def get_prompt_execution_settings_class(self) -> type["PromptExecutionSettings"]:
        """Get the prompt execution settings class."""
        return VoyageAIContextualizedEmbeddingPromptExecutionSettings
