# Copyright (c) Microsoft. All rights reserved.

import sys
from typing import TYPE_CHECKING, Any

from numpy import array, ndarray

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.connectors.ai.voyage_ai.services.voyage_ai_base import VoyageAIBase
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
    """VoyageAI by MongoDB Text Embedding Service.

    Supports current models such as:
    - voyage-4-large
    - voyage-4
    - voyage-4-lite
    - voyage-code-4
    - voyage-finance-2
    - voyage-law-2
    - voyage-4-nano (open-weight)

    Legacy models (e.g. voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3)
    remain accessible from the API.
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
        """Initialize VoyageAI text embedding service.

        Args:
            ai_model_id: The VoyageAI model ID. Falls back to the ``embedding_model_id``
                setting (env var ``VOYAGE_AI_EMBEDDING_MODEL_ID``) when not provided.
            service_id: The service ID (optional).
            api_key: The VoyageAI API key (optional).
            client: A pre-configured VoyageAI client (optional).
            env_file_path: Path to .env file (optional).
            endpoint: VoyageAI API endpoint (optional).
        """
        # Use embedding model from settings if not provided
        if not ai_model_id:
            from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_settings import VoyageAISettings

            settings = VoyageAISettings.create(env_file_path=env_file_path)
            ai_model_id = settings.embedding_model_id

        if not ai_model_id:
            raise ServiceInitializationError(
                "No model ID provided. Set ai_model_id parameter or VOYAGE_AI_EMBEDDING_MODEL_ID environment variable."
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
            texts: List of texts to generate embeddings for (max 1,000 items).
            settings: Prompt execution settings (optional).
            kwargs: Additional arguments to pass to the request.

        Returns:
            ndarray: Array of embeddings.
        """
        if not settings:
            settings = VoyageAIEmbeddingPromptExecutionSettings()
        else:
            settings = self.get_prompt_execution_settings_from_settings(settings)

        try:
            # Call VoyageAI embeddings API
            response = await self.aclient.embed(
                texts=texts,
                model=self.ai_model_id,
                **settings.prepare_settings_dict(),
            )

            # Extract embeddings
            embeddings = response.embeddings
            return array(embeddings)

        except Exception as e:
            raise ServiceResponseException(f"VoyageAI text embedding request failed: {e}") from e

    @override
    def get_prompt_execution_settings_class(self) -> type["PromptExecutionSettings"]:
        """Get the prompt execution settings class."""
        return VoyageAIEmbeddingPromptExecutionSettings
