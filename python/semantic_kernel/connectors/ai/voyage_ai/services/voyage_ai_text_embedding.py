# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

from numpy import array, ndarray

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.connectors.ai.voyage_ai.services.voyage_ai_base import VoyageAIBase
from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_batching import (
    MAX_TOKENS_PER_REQUEST,
    count_tokens_per_text,
    split_into_batches,
)
from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_prompt_execution_settings import (
    VoyageAIEmbeddingPromptExecutionSettings,
)
from semantic_kernel.exceptions.service_exceptions import (
    ServiceInitializationError,
    ServiceResponseException,
)

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings


class VoyageAITextEmbedding(VoyageAIBase, EmbeddingGeneratorBase):
    """Voyage AI by MongoDB Text Embedding Service.

    Supported models (see ``MAX_TOKENS_PER_REQUEST``):
    - voyage-4-large, voyage-4, voyage-4-lite, voyage-code-4
    - voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3
    - voyage-finance-2, voyage-law-2

    voyage-4*, voyage-code-4 and voyage-3* models produce 1024 dimensional embeddings by default and
    support ``output_dimension`` of 2048, 1024, 512 or 256 (Matryoshka). voyage-finance-2 and
    voyage-law-2 produce fixed 1024 dimensional embeddings.

    The inputs are split into requests that respect the model's per-request token limit and the
    1,000 inputs limit; tokens are counted with the tokenizer of the ``voyageai`` package.
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
        """Initialize Voyage AI text embedding service.

        Args:
            ai_model_id: The Voyage AI model ID. Falls back to the ``embedding_model_id``
                setting (env var ``VOYAGE_EMBEDDING_MODEL_ID``) when not provided.
            service_id: The service ID (optional).
            api_key: The Voyage AI API key (optional, defaults to the ``VOYAGE_API_KEY`` env var).
            client: A pre-configured ``voyageai.AsyncClient`` (optional).
            env_file_path: Path to .env file (optional).
            endpoint: Explicit API base URL (optional, overrides the default derived from the API key).
        """
        # Use embedding model from settings if not provided
        if not ai_model_id:
            from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_settings import VoyageAISettings

            settings = VoyageAISettings(env_file_path=env_file_path)  # type: ignore[call-arg]
            ai_model_id = settings.embedding_model_id

        if not ai_model_id:
            raise ServiceInitializationError(
                "No model ID provided. Set ai_model_id parameter or VOYAGE_EMBEDDING_MODEL_ID environment variable."
            )

        if ai_model_id not in MAX_TOKENS_PER_REQUEST:
            raise ServiceInitializationError(
                f"Unknown Voyage AI text embedding model '{ai_model_id}'. "
                f"Supported models: {', '.join(sorted(MAX_TOKENS_PER_REQUEST))}."
            )

        super().__init__(
            ai_model_id=ai_model_id,
            service_id=service_id,
            api_key=api_key,
            client=client,
            env_file_path=env_file_path,
            endpoint=endpoint,
        )

    @override
    async def generate_embeddings(
        self,
        texts: list[str],
        settings: "PromptExecutionSettings | None" = None,
        **kwargs: Any,
    ) -> ndarray:
        """Generate embeddings for the given texts.

        Args:
            texts: List of texts to generate embeddings for. Any number of texts is accepted; they are
                split into requests that respect the model's token and 1,000 inputs limits.
            settings: Prompt execution settings (optional).
            kwargs: Additional arguments to pass to the request.

        Returns:
            ndarray: Array of embeddings, in the order of ``texts``.
        """
        if not settings:
            settings = VoyageAIEmbeddingPromptExecutionSettings()
        else:
            settings = self.get_prompt_execution_settings_from_settings(settings)

        if not texts:
            return array([])

        try:
            max_tokens = MAX_TOKENS_PER_REQUEST[self.ai_model_id]
            if max_tokens is None:
                token_counts = [0] * len(texts)
            else:
                token_counts = await asyncio.to_thread(
                    count_tokens_per_text,
                    partial(self.aclient.tokenize, model=self.ai_model_id),
                    list(texts),
                )

            embeddings: list[Any] = []
            for batch in split_into_batches(texts, token_counts, max_tokens):
                response = await self.aclient.embed(
                    texts=[texts[i] for i in batch],
                    model=self.ai_model_id,
                    **settings.prepare_settings_dict(),
                )
                embeddings.extend(response.embeddings)
            return array(embeddings)

        except Exception as e:
            raise ServiceResponseException(f"Voyage AI text embedding request failed: {e}") from e

    @override
    def get_prompt_execution_settings_class(self) -> type["PromptExecutionSettings"]:
        """Get the prompt execution settings class."""
        return VoyageAIEmbeddingPromptExecutionSettings
