# Copyright (c) Microsoft. All rights reserved.

"""Token-aware request batching for Voyage AI by MongoDB text embedding models.

The embeddings endpoint limits every request to 1,000 inputs and to a model specific total number of
tokens (https://docs.voyageai.com/docs/embeddings). Token counts come from the official ``voyageai``
package tokenizer, they are not estimated.
"""

from collections.abc import Callable, Sequence

MAX_INPUTS_PER_REQUEST = 1000

# Maximum total tokens per embeddings request, per model. ``None`` means the Voyage AI documentation
# lists no per-request token cap for that model, so only MAX_INPUTS_PER_REQUEST applies.
# Values as documented on 2026-11-12.
MAX_TOKENS_PER_REQUEST: dict[str, int | None] = {
    # 1M tokens
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    # 320K tokens
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    # 120K tokens
    "voyage-4-large": 120_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    # No documented per-request token cap
    "voyage-code-4": None,
}


def split_into_batches(
    texts: Sequence[str],
    token_counts: Sequence[int],
    max_tokens: int | None,
    max_inputs: int = MAX_INPUTS_PER_REQUEST,
) -> list[list[int]]:
    """Group input positions into consecutive batches that respect the request limits.

    Batches keep the original input order, so concatenating the results of the batches in order gives the
    results in the order of ``texts``. A single input whose token count alone exceeds ``max_tokens`` gets a
    batch of its own; the service decides what to do with it (``truncation``).

    Args:
        texts: The inputs.
        token_counts: Token count of each input, same length as ``texts``.
        max_tokens: Maximum total tokens per request, or None when the model has no documented cap.
        max_inputs: Maximum number of inputs per request.

    Returns:
        Lists of indexes into ``texts``, one list per request.
    """
    if len(texts) != len(token_counts):
        raise ValueError("texts and token_counts must have the same length.")

    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index, tokens in enumerate(token_counts):
        over_inputs = len(current) >= max_inputs
        over_tokens = max_tokens is not None and current and current_tokens + tokens > max_tokens
        if over_inputs or over_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(index)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def count_tokens_per_text(tokenize: Callable[[list[str]], list], texts: list[str]) -> list[int]:
    """Count the tokens of each text with the ``voyageai`` tokenizer.

    Args:
        tokenize: A bound ``voyageai`` client ``tokenize`` function, already set to the model.
        texts: The inputs.

    Returns:
        The token count of each input.
    """
    return [len(encoding.ids) for encoding in tokenize(texts)]
