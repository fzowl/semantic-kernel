// Copyright (c) Microsoft. All rights reserved.

using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.Connectors.VoyageAI.Core;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Services;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI;

/// <summary>
/// VoyageAI by MongoDB multimodal embedding generation service.
/// Generates embeddings for text, images, or interleaved text and images.
/// Supports the current voyage-multimodal-3.5 model (legacy: voyage-multimodal-3).
/// The voyage-multimodal-3.5 model adds video support in addition to text and images.
/// </summary>
/// <remarks>
/// Constraints:
/// - Maximum 1,000 inputs per request
/// - Images: ≤16 million pixels, ≤20 MB
/// - Total tokens per input: ≤32,000 (560 pixels = 1 token)
/// - Aggregate tokens across inputs: ≤320,000
/// </remarks>
[Experimental("SKEXP0001")]
public sealed class VoyageAIMultimodalEmbeddingGenerationService : ITextEmbeddingGenerationService
{
    private readonly VoyageAIClient _client;
    private readonly string _modelId;
    private readonly Dictionary<string, object?> _attributes = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="VoyageAIMultimodalEmbeddingGenerationService"/> class.
    /// </summary>
    /// <param name="modelId">The VoyageAI by MongoDB model ID (e.g., "voyage-multimodal-3.5", "voyage-multimodal-3").</param>
    /// <param name="apiKey">The VoyageAI by MongoDB API key.</param>
    /// <param name="endpoint">Optional API endpoint. When not set, it is derived from the API key:
    /// keys starting with "al-" use https://ai.mongodb.com/v1, otherwise https://api.voyageai.com/v1.</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <param name="loggerFactory">Optional logger factory.</param>
    public VoyageAIMultimodalEmbeddingGenerationService(
        string modelId,
        string apiKey,
        string? endpoint = null,
        HttpClient? httpClient = null,
        ILoggerFactory? loggerFactory = null)
    {
        Verify.NotNullOrWhiteSpace(modelId);
        Verify.NotNullOrWhiteSpace(apiKey);

        this._modelId = modelId;
        this._client = new VoyageAIClient(
            apiKey,
            endpoint,
            httpClient,
            loggerFactory?.CreateLogger(typeof(VoyageAIMultimodalEmbeddingGenerationService)));

        this._attributes.Add(AIServiceExtensions.ModelIdKey, modelId);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object?> Attributes => this._attributes;

    /// <summary>
    /// Generates multimodal embeddings for interleaved text and/or images.
    /// </summary>
    /// <param name="inputs">List of inputs; each input is composed of one or more ordered
    /// <see cref="VoyageAIMultimodalContent"/> items (text and/or images) and produces one embedding.</param>
    /// <param name="executionSettings">Optional execution settings. Use
    /// <see cref="VoyageAIMultimodalEmbeddingPromptExecutionSettings"/> to control <c>input_type</c>
    /// ("query"/"document") and <c>truncation</c>.</param>
    /// <param name="kernel">The <see cref="Kernel"/> containing services, plugins, and other state.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests.</param>
    /// <returns>A list of multimodal embeddings, one per input.</returns>
    public async Task<IList<ReadOnlyMemory<float>>> GenerateMultimodalEmbeddingsAsync(
        IList<VoyageAIMultimodalInput> inputs,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        Verify.NotNull(inputs);

        var settings = VoyageAIMultimodalEmbeddingPromptExecutionSettings.FromExecutionSettings(executionSettings)
            ?? new VoyageAIMultimodalEmbeddingPromptExecutionSettings();

        var request = new MultimodalEmbeddingRequest
        {
            Inputs = inputs.Select(ToRequestInput).ToList(),
            Model = this._modelId,
            InputType = settings.InputType,
            Truncation = settings.Truncation
        };

        var response = await this._client.SendRequestAsync<MultimodalEmbeddingResponse>(
            "multimodalembeddings",
            request,
            cancellationToken).ConfigureAwait(false);

        var embeddings = response.Data
            .OrderBy(d => d.Index)
            .Select(d => new ReadOnlyMemory<float>(d.Embedding))
            .ToList();

        return embeddings;
    }

    /// <inheritdoc/>
    public Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
        IList<string> data,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
        => this.GenerateEmbeddingsAsync(data, executionSettings: null, kernel, cancellationToken);

    /// <summary>
    /// Generates embeddings for text-only inputs, wrapping each string as a single-item multimodal input.
    /// </summary>
    /// <param name="data">The text inputs to embed.</param>
    /// <param name="executionSettings">Optional execution settings (see
    /// <see cref="VoyageAIMultimodalEmbeddingPromptExecutionSettings"/>).</param>
    /// <param name="kernel">The <see cref="Kernel"/> containing services, plugins, and other state.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests.</param>
    /// <returns>A list of embeddings, one per input text.</returns>
    public Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
        IList<string> data,
        PromptExecutionSettings? executionSettings,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        Verify.NotNull(data);

        // Each text becomes a single-item input, matching the structured payload the API expects.
        var inputs = data.Select(VoyageAIMultimodalInput.FromText).ToList();
        return this.GenerateMultimodalEmbeddingsAsync(inputs, executionSettings, kernel, cancellationToken);
    }

    private static MultimodalInput ToRequestInput(VoyageAIMultimodalInput input)
    {
        Verify.NotNull(input);
        return new MultimodalInput
        {
            Content = input.Content.Select(c => new MultimodalContentItem
            {
                Type = c.Type,
                Text = c.Text,
                ImageUrl = c.ImageUrl,
                ImageBase64 = c.ImageBase64
            }).ToList()
        };
    }
}
