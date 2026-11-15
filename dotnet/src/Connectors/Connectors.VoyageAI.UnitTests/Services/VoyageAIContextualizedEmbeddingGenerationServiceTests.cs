// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Xunit;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI.UnitTests.Services;

/// <summary>
/// Unit tests for <see cref="VoyageAIContextualizedEmbeddingGenerationService"/>.
/// </summary>
public sealed class VoyageAIContextualizedEmbeddingGenerationServiceTests : IDisposable
{
    private readonly HttpMessageHandlerStub _messageHandlerStub;
    private readonly HttpClient _httpClient;

    public VoyageAIContextualizedEmbeddingGenerationServiceTests()
    {
        this._messageHandlerStub = new HttpMessageHandlerStub();
        this._httpClient = new HttpClient(this._messageHandlerStub);
    }

    [Fact]
    public void Constructor_WithValidParameters_CreatesInstance()
    {
        // Arrange & Act
        var service = new VoyageAIContextualizedEmbeddingGenerationService(
            modelId: "voyage-context-4",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        // Assert
        service.Should().NotBeNull();
        service.Attributes.Should().ContainKey("ModelId");
        service.Attributes["ModelId"].Should().Be("voyage-context-4");
    }

    [Fact]
    public void Constructor_WithNullModelId_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new VoyageAIContextualizedEmbeddingGenerationService(
                modelId: null!,
                apiKey: "test-api-key"
            )
        );
    }

    [Fact]
    public void Constructor_WithNullApiKey_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new VoyageAIContextualizedEmbeddingGenerationService(
                modelId: "voyage-context-4",
                apiKey: null!
            )
        );
    }

    [Fact]
    public async Task DocumentModeSendsFlatDocumentsWithAutoChunkingAsync()
    {
        // Arrange
        this.SetResponse([[("chunk a", 0.1f), ("chunk b", 0.2f)], [("chunk c", 0.3f)]]);
        var service = this.CreateService();

        // Act
        var embeddings = await service.GenerateContextualizedEmbeddingsAsync(["doc one", "doc two"]);

        // Assert
        using var request = JsonDocument.Parse(this._messageHandlerStub.RequestContent!);
        var root = request.RootElement;
        root.GetProperty("inputs").EnumerateArray().Select(e => e.GetString()).Should().Equal("doc one", "doc two");
        root.GetProperty("input_type").GetString().Should().Be("document");
        root.GetProperty("enable_auto_chunking").GetBoolean().Should().BeTrue();
        root.GetProperty("chunk_size").GetInt32().Should().Be(32_000);
        root.TryGetProperty("truncation", out _).Should().BeFalse();
        this._messageHandlerStub.RequestUri!.ToString().Should().EndWith("/contextualizedembeddings");
        embeddings.Should().HaveCount(3);
    }

    [Fact]
    public async Task ChunkEmbeddingsReturnChunkTextsPerDocumentAsync()
    {
        // Arrange
        this.SetResponse([[("chunk a", 0.1f), ("chunk b", 0.2f)], [("chunk c", 0.3f)]]);
        var service = this.CreateService();

        // Act
        var results = await service.GenerateContextualizedChunkEmbeddingsAsync(["doc one", "doc two"]);

        // Assert
        results.Should().HaveCount(2);
        results[0].Index.Should().Be(0);
        results[0].ChunkTexts.Should().Equal("chunk a", "chunk b");
        results[0].Embeddings.Should().HaveCount(2);
        results[1].ChunkTexts.Should().Equal("chunk c");
    }

    [Fact]
    public async Task QueryModeWrapsEachQueryAndSkipsAutoChunkingAsync()
    {
        // Arrange
        this.SetResponse([[(null, 0.1f)], [(null, 0.2f)]]);
        var service = this.CreateService();
        var settings = new VoyageAIContextualizedEmbeddingPromptExecutionSettings { InputType = "query" };

        // Act
        var embeddings = await service.GenerateContextualizedEmbeddingsAsync(["q1", "q2"], settings);

        // Assert
        using var request = JsonDocument.Parse(this._messageHandlerStub.RequestContent!);
        var root = request.RootElement;
        root.GetProperty("inputs").GetRawText().Should().Be("[[\"q1\"],[\"q2\"]]");
        root.GetProperty("input_type").GetString().Should().Be("query");
        root.TryGetProperty("enable_auto_chunking", out _).Should().BeFalse();
        root.TryGetProperty("chunk_size", out _).Should().BeFalse();
        embeddings.Should().HaveCount(2);
    }

    [Fact]
    public async Task UnsupportedInputTypeIsRejectedBeforeTheRequestAsync()
    {
        // Arrange
        var service = this.CreateService();
        var settings = new VoyageAIContextualizedEmbeddingPromptExecutionSettings { InputType = "other" };

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => service.GenerateContextualizedEmbeddingsAsync(["doc"], settings));
        this._messageHandlerStub.RequestContent.Should().BeNull();
    }

    [Fact]
    public async Task GenerateEmbeddingsTreatsEachTextAsADocumentAsync()
    {
        // Arrange
        this.SetResponse([[("a", 0.1f)], [("b", 0.2f)]]);
        var service = this.CreateService();

        // Act
        var embeddings = await service.GenerateEmbeddingsAsync(["a", "b"]);

        // Assert
        using var request = JsonDocument.Parse(this._messageHandlerStub.RequestContent!);
        request.RootElement.GetProperty("inputs").EnumerateArray().Select(e => e.GetString()).Should().Equal("a", "b");
        embeddings.Should().HaveCount(2);
    }

    [Fact]
    public async Task EmptyInputMakesNoRequestAsync()
    {
        // Arrange
        var service = this.CreateService();

        // Act
        var embeddings = await service.GenerateContextualizedEmbeddingsAsync([]);

        // Assert
        embeddings.Should().BeEmpty();
        this._messageHandlerStub.RequestContent.Should().BeNull();
    }

    [Fact]
    public async Task ApiErrorSurfacesTheDetailTextAsync()
    {
        // Arrange
        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.BadRequest)
        {
            Content = new StringContent("{\"detail\":\"Model nope is not supported.\"}")
        };
        var service = this.CreateService();

        // Act
        var exception = await Assert.ThrowsAsync<HttpRequestException>(() => service.GenerateContextualizedEmbeddingsAsync(["doc"]));

        // Assert
        exception.Message.Should().Contain("Model nope is not supported.").And.NotContain("{");
        exception.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    private VoyageAIContextualizedEmbeddingGenerationService CreateService()
        => new(modelId: "voyage-context-4", apiKey: "test-api-key", httpClient: this._httpClient);

    private void SetResponse((string? Text, float Value)[][] documents)
    {
        var responseContent = JsonSerializer.Serialize(new
        {
            data = documents.Select((chunks, documentIndex) => new
            {
                data = chunks.Select((chunk, chunkIndex) => new { embedding = new[] { chunk.Value }, index = chunkIndex, text = chunk.Text }),
                index = documentIndex,
            }),
            usage = new { total_tokens = 5 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };
    }

    public void Dispose()
    {
        this._httpClient.Dispose();
        this._messageHandlerStub.Dispose();
    }

    private sealed class HttpMessageHandlerStub : HttpMessageHandler
    {
        public HttpResponseMessage? ResponseToReturn { get; set; }
        public string? RequestContent { get; private set; }
        public Uri? RequestUri { get; private set; }
        public Dictionary<string, string> RequestHeaders { get; } = new();

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            this.RequestUri = request.RequestUri;
            if (request.Content is not null)
            {
                this.RequestContent = await request.Content.ReadAsStringAsync(cancellationToken);
            }

            foreach (var header in request.Headers)
            {
                this.RequestHeaders[header.Key] = string.Join(",", header.Value);
            }

            return this.ResponseToReturn ?? new HttpResponseMessage(HttpStatusCode.OK);
        }
    }
}
