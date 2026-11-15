// Copyright (c) Microsoft. All rights reserved.

using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.Connectors.VoyageAI.Core;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Services;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI;

/// <summary>
/// Voyage AI by MongoDB contextualized embedding generation service.
/// Generates embeddings that capture both local chunk details and global document-level metadata
/// using the voyage-context-4 or voyage-context-3 model.
/// </summary>
[Experimental("SKEXP0001")]
public sealed class VoyageAIContextualizedEmbeddingGenerationService : ITextEmbeddingGenerationService
{
    private const int AutoChunkSize = 32_000;

    private readonly VoyageAIClient _client;
    private readonly string _modelId;
    private readonly Dictionary<string, object?> _attributes = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="VoyageAIContextualizedEmbeddingGenerationService"/> class.
    /// </summary>
    /// <param name="modelId">The Voyage AI by MongoDB model ID (e.g., voyage-context-4).</param>
    /// <param name="apiKey">The Voyage AI by MongoDB API key.</param>
    /// <param name="endpoint">Optional API endpoint. When not set, it is derived from the API key:
    /// keys starting with "al-" use https://ai.mongodb.com/v1, otherwise https://api.voyageai.com/v1.</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <param name="loggerFactory">Optional logger factory.</param>
    public VoyageAIContextualizedEmbeddingGenerationService(
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
            loggerFactory?.CreateLogger(typeof(VoyageAIContextualizedEmbeddingGenerationService)));

        this._attributes.Add(AIServiceExtensions.ModelIdKey, modelId);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object?> Attributes => this._attributes;

    /// <summary>
    /// Generates contextualized embeddings.
    /// In "document" mode (the default) every string is one full document: the API chunks it
    /// (auto-chunking, chunk size 32,000) and returns one embedding per chunk, documents in input order.
    /// In "query" mode every string is one query and the result has one embedding per query.
    /// </summary>
    /// <param name="inputs">One document or one query per element. Pre-chunked input is not accepted:
    /// the API rejects it with auto-chunking.</param>
    /// <param name="executionSettings">Optional execution settings. Use
    /// <see cref="VoyageAIContextualizedEmbeddingPromptExecutionSettings"/> to control <c>input_type</c>
    /// ("document"/"query").</param>
    /// <param name="kernel">The <see cref="Kernel"/> containing services, plugins, and other state.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests.</param>
    /// <returns>A list of embeddings for all chunks across all documents.</returns>
    public async Task<IList<ReadOnlyMemory<float>>> GenerateContextualizedEmbeddingsAsync(
        IList<string> inputs,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var results = await this.GenerateContextualizedChunkEmbeddingsAsync(inputs, executionSettings, kernel, cancellationToken).ConfigureAwait(false);

        return results.SelectMany(r => r.Embeddings).ToList();
    }

    /// <summary>
    /// Generates contextualized embeddings and returns them per input, together with the chunk texts.
    /// </summary>
    /// <param name="inputs">One document or one query per element.</param>
    /// <param name="executionSettings">Optional execution settings (see
    /// <see cref="VoyageAIContextualizedEmbeddingPromptExecutionSettings"/>).</param>
    /// <param name="kernel">The <see cref="Kernel"/> containing services, plugins, and other state.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests.</param>
    /// <returns>One result per input, in input order.</returns>
    public async Task<IList<VoyageAIContextualizedChunkEmbeddings>> GenerateContextualizedChunkEmbeddingsAsync(
        IList<string> inputs,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        Verify.NotNull(inputs);

        var settings = VoyageAIContextualizedEmbeddingPromptExecutionSettings.FromExecutionSettings(executionSettings)
            ?? new VoyageAIContextualizedEmbeddingPromptExecutionSettings();

        if (inputs.Count == 0)
        {
            return [];
        }

        ContextualizedEmbeddingRequest request;
        switch (settings.InputType)
        {
            case "document":
                request = new ContextualizedEmbeddingRequest
                {
                    Inputs = inputs.Cast<object>().ToList(),
                    Model = this._modelId,
                    InputType = "document",
                    // Auto-chunking only works with input_type "document" and one string per document.
                    EnableAutoChunking = true,
                    ChunkSize = AutoChunkSize,
                };
                break;
            case "query":
                // A query is a document of one chunk; there is no auto-chunking for queries.
                request = new ContextualizedEmbeddingRequest
                {
                    Inputs = inputs.Select(query => (object)new List<string> { query }).ToList(),
                    Model = this._modelId,
                    InputType = "query",
                };
                break;
            default:
                throw new ArgumentException(
                    $"Unsupported input type '{settings.InputType}', expected 'document' or 'query'.",
                    nameof(executionSettings));
        }

        var response = await this._client.SendRequestAsync<ContextualizedEmbeddingResponse>(
            "contextualizedembeddings",
            request,
            cancellationToken).ConfigureAwait(false);

        return response.Data
            .OrderBy(document => document.Index)
            .Select(document =>
            {
                var items = document.Data.OrderBy(item => item.Index).ToList();
                return new VoyageAIContextualizedChunkEmbeddings(
                    document.Index,
                    items.Where(item => item.Text is not null).Select(item => item.Text!).ToList(),
                    items.Select(item => new ReadOnlyMemory<float>(item.Embedding)).ToList());
            })
            .ToList();
    }

    /// <inheritdoc/>
    public Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
        IList<string> data,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
        => this.GenerateEmbeddingsAsync(data, executionSettings: null, kernel, cancellationToken);

    /// <summary>
    /// Generates embeddings for the given texts, each text being one document
    /// (or one query when the settings select the "query" input type).
    /// </summary>
    /// <param name="data">The documents or queries to embed.</param>
    /// <param name="executionSettings">Optional execution settings (see
    /// <see cref="VoyageAIContextualizedEmbeddingPromptExecutionSettings"/>).</param>
    /// <param name="kernel">The <see cref="Kernel"/> containing services, plugins, and other state.</param>
    /// <param name="cancellationToken">The <see cref="CancellationToken"/> to monitor for cancellation requests.</param>
    /// <returns>A list of embeddings, one per input.</returns>
    /// <exception cref="InvalidOperationException">The API split an input into several chunks.</exception>
    public async Task<IList<ReadOnlyMemory<float>>> GenerateEmbeddingsAsync(
        IList<string> data,
        PromptExecutionSettings? executionSettings,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var results = await this.GenerateContextualizedChunkEmbeddingsAsync(data, executionSettings, kernel, cancellationToken).ConfigureAwait(false);

        if (results.Any(result => result.Embeddings.Count != 1))
        {
            throw new InvalidOperationException(
                "GenerateEmbeddingsAsync returns one embedding per input, but the API split at least one input into several chunks. " +
                $"Use {nameof(this.GenerateContextualizedChunkEmbeddingsAsync)} to get the embeddings per chunk.");
        }

        return results.Select(result => result.Embeddings[0]).ToList();
    }
}

/// <summary>
/// The contextualized embeddings of one input.
/// </summary>
/// <param name="Index">Position of the input in the request.</param>
/// <param name="ChunkTexts">The chunks the API split the input into, empty when the API returned no chunk texts.</param>
/// <param name="Embeddings">One embedding per chunk, in the order of <paramref name="ChunkTexts"/>.</param>
[Experimental("SKEXP0001")]
public sealed record VoyageAIContextualizedChunkEmbeddings(
    int Index,
    IReadOnlyList<string> ChunkTexts,
    IReadOnlyList<ReadOnlyMemory<float>> Embeddings);
