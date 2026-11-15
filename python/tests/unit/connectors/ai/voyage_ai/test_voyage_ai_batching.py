# Copyright (c) Microsoft. All rights reserved.

import pytest

from semantic_kernel.connectors.ai.voyage_ai.voyage_ai_batching import (
    MAX_TOKENS_PER_REQUEST,
    count_tokens_per_text,
    split_into_batches,
)


def test_split_respects_token_limit():
    assert split_into_batches(["a"] * 4, [4, 4, 4, 4], max_tokens=8) == [[0, 1], [2, 3]]


def test_split_respects_input_limit():
    assert split_into_batches(["a"] * 5, [1] * 5, max_tokens=None, max_inputs=2) == [[0, 1], [2, 3], [4]]


def test_oversized_single_input_gets_its_own_batch():
    assert split_into_batches(["a"] * 3, [1, 50, 1], max_tokens=10) == [[0], [1], [2]]


def test_empty_input_gives_no_batches():
    assert split_into_batches([], [], max_tokens=10) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        split_into_batches(["a"], [1, 2], max_tokens=10)


def test_count_tokens_uses_tokenizer_ids():
    class _Encoding:
        def __init__(self, ids):
            self.ids = ids

    assert count_tokens_per_text(lambda texts: [_Encoding([1, 2, 3]), _Encoding([1])], ["x", "y"]) == [3, 1]


@pytest.mark.parametrize(
    ("model", "limit"),
    [("voyage-4-lite", 1_000_000), ("voyage-4", 320_000), ("voyage-3.5", 320_000), ("voyage-4-large", 120_000),
     ("voyage-code-3", 120_000), ("voyage-finance-2", 120_000), ("voyage-law-2", 120_000), ("voyage-code-4", None)],
)  # fmt: skip
def test_documented_model_limits(model, limit):
    assert MAX_TOKENS_PER_REQUEST[model] == limit
