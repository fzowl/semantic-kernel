# Copyright (c) Microsoft. All rights reserved.

from typing import ClassVar

from pydantic import SecretStr

from semantic_kernel.kernel_pydantic import KernelBaseSettings


class VoyageAISettings(KernelBaseSettings):
    """VoyageAI by MongoDB service settings.

    Note:
        The official ``voyageai`` package automatically routes requests based on the
        API key prefix: keys starting with ``al-`` are sent to ``https://ai.mongodb.com/v1``,
        all other keys are sent to ``https://api.voyageai.com/v1``. The ``endpoint`` value
        below is only used for reporting and as an explicit override.

    Args:
        api_key: VoyageAI by MongoDB API key (Env var VOYAGE_AI_API_KEY)
        embedding_model_id: Default embedding model ID
            (Env var VOYAGE_AI_EMBEDDING_MODEL_ID)
        contextualized_embedding_model_id: Contextualized embedding model ID
            (Env var VOYAGE_AI_CONTEXTUALIZED_EMBEDDING_MODEL_ID)
        multimodal_embedding_model_id: Multimodal embedding model ID
            (Env var VOYAGE_AI_MULTIMODAL_EMBEDDING_MODEL_ID)
        reranker_model_id: Reranker model ID (Env var VOYAGE_AI_RERANKER_MODEL_ID)
        endpoint: VoyageAI API endpoint (Env var VOYAGE_AI_ENDPOINT),
            defaults to https://api.voyageai.com/v1
        max_retries: Maximum number of retries for API calls
            (Env var VOYAGE_AI_MAX_RETRIES), defaults to 3
        env_file_path: Path to .env file (optional)
    """

    env_prefix: ClassVar[str] = "VOYAGE_AI_"

    api_key: SecretStr
    embedding_model_id: str | None = None
    contextualized_embedding_model_id: str | None = None
    multimodal_embedding_model_id: str | None = None
    reranker_model_id: str | None = None
    endpoint: str = "https://api.voyageai.com/v1"
    max_retries: int = 3
