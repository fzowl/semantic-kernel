# Copyright (c) Microsoft. All rights reserved.

from typing import ClassVar

from pydantic import SecretStr

from semantic_kernel.kernel_pydantic import KernelBaseSettings


class VoyageAISettings(KernelBaseSettings):
    """Voyage AI by MongoDB service settings.

    Note:
        The official ``voyageai`` package selects the API host from the API key prefix:
        keys starting with ``al-`` are sent to ``https://ai.mongodb.com/v1``, all other
        keys are sent to ``https://api.voyageai.com/v1``. An explicitly set ``endpoint``
        always takes precedence over that default.

    Args:
        api_key: Voyage AI by MongoDB API key (Env var VOYAGE_API_KEY)
        embedding_model_id: Text embedding model ID (Env var VOYAGE_EMBEDDING_MODEL_ID)
        contextualized_embedding_model_id: Contextualized embedding model ID
            (Env var VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL_ID)
        multimodal_embedding_model_id: Multimodal embedding model ID
            (Env var VOYAGE_MULTIMODAL_EMBEDDING_MODEL_ID)
        reranker_model_id: Reranker model ID (Env var VOYAGE_RERANKER_MODEL_ID)
        endpoint: Explicit API base URL (Env var VOYAGE_ENDPOINT). When not set, the
            default is derived from the API key prefix as described above.
        max_retries: Maximum number of retries for API calls (Env var VOYAGE_MAX_RETRIES),
            defaults to 3
        env_file_path: Path to .env file (optional)
    """

    env_prefix: ClassVar[str] = "VOYAGE_"

    api_key: SecretStr
    embedding_model_id: str | None = None
    contextualized_embedding_model_id: str | None = None
    multimodal_embedding_model_id: str | None = None
    reranker_model_id: str | None = None
    endpoint: str | None = None
    max_retries: int = 3
