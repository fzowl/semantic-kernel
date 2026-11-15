# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Literal

from pydantic import Field

from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings


class VoyageAIEmbeddingPromptExecutionSettings(PromptExecutionSettings):
    """Voyage AI by MongoDB embedding prompt execution settings.

    Args:
        ai_model_id: The model ID to use for embeddings.
        input_type: Type of input ('query', 'document', or None).
        truncation: Whether to truncate inputs that exceed the model's context length.
        output_dimension: Desired embedding dimension (256, 512, 1024, or 2048), only sent when set.
            Supported by the voyage-4*, voyage-code-4 and voyage-3* models.
        output_dtype: Output data type ('float', 'int8', 'uint8', 'binary', or 'ubinary').
    """

    ai_model_id: str | None = None
    input_type: Literal["query", "document"] | None = None
    truncation: bool = True
    output_dimension: int | None = None
    output_dtype: Literal["float", "int8", "uint8", "binary", "ubinary"] | None = Field(
        default=None, alias="encoding_format"
    )

    def prepare_settings_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare settings for API call."""
        settings: dict[str, Any] = {}
        if self.input_type:
            settings["input_type"] = self.input_type
        if self.truncation is not None:
            settings["truncation"] = self.truncation
        if self.output_dimension:
            settings["output_dimension"] = self.output_dimension
        if self.output_dtype:
            settings["output_dtype"] = self.output_dtype
        return settings


class VoyageAIContextualizedEmbeddingPromptExecutionSettings(PromptExecutionSettings):
    """Voyage AI by MongoDB contextualized embedding prompt execution settings.

    Args:
        ai_model_id: The model ID to use for contextualized embeddings.
        input_type: 'document' (every input is a full document that the API chunks and embeds, default)
            or 'query' (every input is one query).
    """

    ai_model_id: str | None = None
    input_type: Literal["query", "document"] = "document"

    def prepare_settings_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare settings for API call."""
        return {"input_type": self.input_type}


class VoyageAIMultimodalEmbeddingPromptExecutionSettings(PromptExecutionSettings):
    """Voyage AI by MongoDB multimodal embedding prompt execution settings.

    Args:
        ai_model_id: The model ID to use for multimodal embeddings.
        input_type: Type of input ('query' or 'document').
        truncation: Whether to truncate inputs that exceed the model's context length.
        output_dimension: Desired embedding dimension (256, 512, 1024, or 2048), only sent when set.
            Supported by voyage-multimodal-3.5.
    """

    ai_model_id: str | None = None
    input_type: Literal["query", "document"] | None = None
    truncation: bool = True
    output_dimension: int | None = None

    def prepare_settings_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare settings for API call."""
        settings: dict[str, Any] = {}
        if self.input_type:
            settings["input_type"] = self.input_type
        if self.truncation is not None:
            settings["truncation"] = self.truncation
        if self.output_dimension:
            settings["output_dimension"] = self.output_dimension
        return settings


class VoyageAIRerankPromptExecutionSettings(PromptExecutionSettings):
    """Voyage AI by MongoDB rerank prompt execution settings.

    Args:
        ai_model_id: The model ID to use for reranking.
        top_k: Number of most relevant documents to return.
        truncation: Whether to truncate inputs that exceed the model's context length.
    """

    ai_model_id: str | None = None
    top_k: int | None = None
    truncation: bool = True

    def prepare_settings_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare settings for API call."""
        settings: dict[str, Any] = {}
        if self.top_k:
            settings["top_k"] = self.top_k
        if self.truncation is not None:
            settings["truncation"] = self.truncation
        return settings
