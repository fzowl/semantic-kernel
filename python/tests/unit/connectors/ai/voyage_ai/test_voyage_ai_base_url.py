# Copyright (c) Microsoft. All rights reserved.

import pytest

from semantic_kernel.connectors.ai.voyage_ai import VoyageAIReranker, VoyageAITextEmbedding


@pytest.mark.parametrize(
    ("api_key", "endpoint", "expected"),
    [
        ("pa-key", None, "https://api.voyageai.com/v1"),
        ("al-key", None, "https://ai.mongodb.com/v1"),
        ("al-key", "https://custom.example/v1", "https://custom.example/v1"),
        ("pa-key", "https://custom.example/v1", "https://custom.example/v1"),
    ],
)
def test_host_is_chosen_by_key_prefix_and_explicit_endpoint_wins(api_key, endpoint, expected):
    service = VoyageAITextEmbedding(ai_model_id="voyage-4", api_key=api_key, endpoint=endpoint)

    assert service.aclient._params["base_url"] == expected


def test_injected_client_is_used():
    from unittest.mock import MagicMock

    client = MagicMock()
    service = VoyageAIReranker(ai_model_id="rerank-3", api_key="pa-key", client=client)

    assert service.aclient is client
