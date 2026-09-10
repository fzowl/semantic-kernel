// Copyright (c) Microsoft. All rights reserved.

using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.Connectors.VoyageAI.Core;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Services;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI;

/// <summary>
/// VoyageAI by MongoDB text embedding generation service.
/// Supports current models such as voyage-4-large, voyage-4, voyage-4-lite, voyage-code-4,
/// voyage-finance-2, and voyage-law-2 (legacy voyage-3.x models remain accessible).
/// </summary>
[Experimental("SKEXP0001")]
public sealed class VoyageAITextEmbeddingGenerationService : ITextEmbeddingGenerationService
{
    private readonly VoyageAIClient _client;
    private readonly string _modelId;
    private readonly Dictionary<string, object?> _attributes = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="VoyageAITextEmbeddingGenerationService"/> class.
    /// </summary>
    /// <param name="modelId">The VoyageAI by MongoDB model ID.</param>
    /// <param name="apiKey">The VoyageAI by MongoDB API key.</param>
    /// <param name="endpoint">Optional API endpoint. When not set, it is derived from the API key:
    /// keys starting with "al-" use https://ai.mongodb.com/v1, otherwise https://api.voyageai.com/v1.</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <param name="loggerFactory">Optional logger factory.</param>
    public VoyageAITextEmbeddingGenerationService(
        string modelId,
        string apiKey,
        string? endpoint = null,
        HttpClient? httpClient = null,
        ILoggerFactory? loggerFactory = null)
    {
        ArgumentNullException.ThrowIfNullOrWhiteSpace(modelId);
        ArgumentNullException.ThrowIfNullOrWhiteSpace(apiKey);

        this._modelId = modelId;
        this._client = new VoyageAIClient(
            apiKey,
            endpoint,
            httpClient,
            loggerFactory?.CreateLogger(typeof(VoyageAITextEmbeddingGenerationService)));

        this._attributes.Add(AIServiceExtensions.ModelIdKey, modelId);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object?> Attributes => this._attributes;

    /// <inheritdoc/>
    public Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
        IList<string> data,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
        => this.GenerateEmbeddingsAsync(data, executionSettings: null, kernel, cancellationToken);

    /// <summary>
    /// Generates embeddings for the given data using the supplied execution settings.
    /// </summary>
    /// <param name="data">The text inputs to embed.</param>
    /// <param name="executionSettings">Optional execution settings. Use
    /// <see cref="VoyageAIEmbeddingPromptExecutionSettings"/> to control <c>input_type</c>
    /// ("query"/"document"), <c>truncation</c>, <c>output_dimension</c> and <c>output_dtype</c>.</param>
    /// <param name="kernel">The <see cref="Kernel"/> containing services, plugins, and other state.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests.</param>
    /// <returns>A list of embeddings, one per input text.</returns>
    public async Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
        IList<string> data,
        PromptExecutionSettings? executionSettings,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        Verify.NotNull(data);

        var settings = VoyageAIEmbeddingPromptExecutionSettings.FromExecutionSettings(executionSettings)
            ?? new VoyageAIEmbeddingPromptExecutionSettings();

        var request = new EmbeddingRequest
        {
            Input = data,
            Model = this._modelId,
            InputType = settings.InputType,
            Truncation = settings.Truncation,
            OutputDimension = settings.OutputDimension,
            OutputDtype = settings.OutputDtype
        };

        var response = await this._client.SendRequestAsync<EmbeddingResponse>(
            "embeddings",
            request,
            cancellationToken).ConfigureAwait(false);

        var embeddings = response.Data
            .OrderBy(d => d.Index)
            .Select(d => new ReadOnlyMemory<float>(d.Embedding))
            .ToList();

        return embeddings;
    }
}
