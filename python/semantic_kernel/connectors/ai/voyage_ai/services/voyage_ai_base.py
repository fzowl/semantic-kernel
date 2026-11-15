# Copyright (c) Microsoft. All rights reserved.

from typing import Any

from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_settings import VoyageAISettings
from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase


class VoyageAIBase(AIServiceClientBase):
    """Base class for Voyage AI by MongoDB services."""

    def __init__(
        self,
        ai_model_id: str,
        service_id: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        env_file_path: str | None = None,
        endpoint: str | None = None,
        max_retries: int | None = None,
    ):
        """Initialize Voyage AI base service.

        Args:
            ai_model_id: The Voyage AI model ID.
            service_id: The service ID (optional, defaults to ai_model_id).
            api_key: The Voyage AI API key (optional, will use the VOYAGE_API_KEY env var if not provided).
            client: A pre-configured ``voyageai.AsyncClient`` (optional).
            env_file_path: Path to .env file (optional).
            endpoint: Explicit API base URL (optional). When not set, the ``voyageai`` package picks the
                default from the API key prefix: ``al-`` keys use https://ai.mongodb.com/v1, all other keys
                use https://api.voyageai.com/v1.
            max_retries: Maximum number of retries for API calls (optional, defaults to 3).
        """
        try:
            voyage_settings = VoyageAISettings(
                api_key=api_key,  # type: ignore[arg-type]
                endpoint=endpoint,
                max_retries=max_retries,  # type: ignore[arg-type]
                env_file_path=env_file_path,
            )
        except Exception as e:
            raise ServiceInitializationError(f"Failed to load Voyage AI settings: {e}") from e

        super().__init__(
            ai_model_id=ai_model_id,
            service_id=service_id or ai_model_id,
        )

        # Private attributes, to avoid Pydantic validation of the client.
        self._voyage_settings = voyage_settings
        self._endpoint = voyage_settings.endpoint
        self._aclient = client or self._create_async_client(voyage_settings)

    @property
    def aclient(self):
        """Get the Voyage AI async client."""
        return self._aclient

    @property
    def endpoint(self) -> str | None:
        """Get the explicitly configured API base URL, None when the SDK default is used."""
        return self._endpoint

    @staticmethod
    def _create_async_client(settings: VoyageAISettings):
        """Create the Voyage AI async client.

        Args:
            settings: Voyage AI settings.

        Returns:
            Voyage AI async client instance.
        """
        try:
            import voyageai
        except ImportError as e:
            raise ServiceInitializationError(
                "The voyageai package is required to use Voyage AI services. "
                "Please install it with: pip install semantic-kernel[voyageai]"
            ) from e

        # The official client sets `base_url = base_url or get_default_base_url(api_key)`: an explicit
        # endpoint wins, otherwise "al-" keys go to https://ai.mongodb.com/v1 and all others to
        # https://api.voyageai.com/v1.
        return voyageai.AsyncClient(
            api_key=settings.api_key.get_secret_value(),
            max_retries=settings.max_retries,
            base_url=settings.endpoint,
        )

    def service_url(self) -> str | None:
        """Get the explicitly configured service URL, None when the SDK default is used."""
        return self.endpoint
