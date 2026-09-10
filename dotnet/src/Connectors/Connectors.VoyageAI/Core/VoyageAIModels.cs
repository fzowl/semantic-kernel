// Copyright (c) Microsoft. All rights reserved.

using System.Text.Json.Serialization;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI.Core;

#region Embedding Models

internal sealed class EmbeddingRequest
{
    [JsonPropertyName("input")]
    public required IList<string> Input { get; set; }

    [JsonPropertyName("model")]
    public required string Model { get; set; }

    [JsonPropertyName("input_type")]
    public string? InputType { get; set; }

    [JsonPropertyName("truncation")]
    public bool? Truncation { get; set; }

    [JsonPropertyName("output_dimension")]
    public int? OutputDimension { get; set; }

    [JsonPropertyName("output_dtype")]
    public string? OutputDtype { get; set; }
}

internal sealed class EmbeddingResponse
{
    [JsonPropertyName("data")]
    public required IList<EmbeddingDataItem> Data { get; set; }

    [JsonPropertyName("usage")]
    public required EmbeddingUsage Usage { get; set; }
}

internal sealed class EmbeddingDataItem
{
    [JsonPropertyName("object")]
    public string? Object { get; set; }

    [JsonPropertyName("embedding")]
    public required float[] Embedding { get; set; }

    [JsonPropertyName("index")]
    public int Index { get; set; }
}

internal sealed class EmbeddingUsage
{
    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}

#endregion

#region Contextualized Embedding Models

internal sealed class ContextualizedEmbeddingRequest
{
    [JsonPropertyName("inputs")]
    public required IList<IList<string>> Inputs { get; set; }

    [JsonPropertyName("model")]
    public required string Model { get; set; }

    [JsonPropertyName("input_type")]
    public string? InputType { get; set; }

    [JsonPropertyName("truncation")]
    public bool? Truncation { get; set; }

    [JsonPropertyName("output_dimension")]
    public int? OutputDimension { get; set; }

    [JsonPropertyName("output_dtype")]
    public string? OutputDtype { get; set; }
}

internal sealed class ContextualizedEmbeddingResponse
{
    [JsonPropertyName("results")]
    public required IList<ContextualizedEmbeddingResult> Results { get; set; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}

internal sealed class ContextualizedEmbeddingResult
{
    [JsonPropertyName("embeddings")]
    public required IList<EmbeddingItem> Embeddings { get; set; }
}

internal sealed class EmbeddingItem
{
    [JsonPropertyName("embedding")]
    public required float[] Embedding { get; set; }

    [JsonPropertyName("chunk")]
    public string? Chunk { get; set; }

    [JsonPropertyName("index")]
    public int Index { get; set; }
}

#endregion

#region Reranking Models

internal sealed class RerankRequest
{
    [JsonPropertyName("query")]
    public required string Query { get; set; }

    [JsonPropertyName("documents")]
    public required IList<string> Documents { get; set; }

    [JsonPropertyName("model")]
    public required string Model { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("truncation")]
    public bool? Truncation { get; set; }
}

internal sealed class RerankResponse
{
    [JsonPropertyName("data")]
    public required IList<RerankDataItem> Data { get; set; }

    [JsonPropertyName("usage")]
    public required EmbeddingUsage Usage { get; set; }
}

internal sealed class RerankDataItem
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("relevance_score")]
    public double RelevanceScore { get; set; }
}

#endregion

#region Multimodal Embedding Models

internal sealed class MultimodalEmbeddingRequest
{
    // Each input is an object with a "content" list, per
    // https://docs.voyageai.com/reference/multimodal-embeddings-api
    [JsonPropertyName("inputs")]
    public required IList<MultimodalInput> Inputs { get; set; }

    [JsonPropertyName("model")]
    public required string Model { get; set; }

    [JsonPropertyName("input_type")]
    public string? InputType { get; set; }

    [JsonPropertyName("truncation")]
    public bool? Truncation { get; set; }
}

internal sealed class MultimodalInput
{
    [JsonPropertyName("content")]
    public required IList<MultimodalContentItem> Content { get; set; }
}

internal sealed class MultimodalContentItem
{
    /// <summary>One of "text", "image_url", or "image_base64".</summary>
    [JsonPropertyName("type")]
    public required string Type { get; set; }

    [JsonPropertyName("text")]
    public string? Text { get; set; }

    [JsonPropertyName("image_url")]
    public string? ImageUrl { get; set; }

    [JsonPropertyName("image_base64")]
    public string? ImageBase64 { get; set; }
}

internal sealed class MultimodalEmbeddingResponse
{
    [JsonPropertyName("data")]
    public required IList<EmbeddingDataItem> Data { get; set; }

    [JsonPropertyName("usage")]
    public required EmbeddingUsage Usage { get; set; }
}

#endregion
