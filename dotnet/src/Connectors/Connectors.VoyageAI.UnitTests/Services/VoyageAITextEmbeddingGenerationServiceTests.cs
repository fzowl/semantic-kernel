// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.SemanticKernel.Connectors.VoyageAI;
using Moq;
using Moq.Protected;
using Xunit;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI.UnitTests.Services;

/// <summary>
/// Unit tests for <see cref="VoyageAITextEmbeddingGenerationService"/>.
/// </summary>
public sealed class VoyageAITextEmbeddingGenerationServiceTests : IDisposable
{
    private readonly HttpMessageHandlerStub _messageHandlerStub;
    private readonly HttpClient _httpClient;

    public VoyageAITextEmbeddingGenerationServiceTests()
    {
        this._messageHandlerStub = new HttpMessageHandlerStub();
        this._httpClient = new HttpClient(this._messageHandlerStub);
    }

    [Fact]
    public void ConstructorShouldInitializeService()
    {
        // Arrange & Act
        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-3-large",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        // Assert
        service.Should().NotBeNull();
        service.Attributes.Should().ContainKey("ModelId");
    }

    [Fact]
    public void ConstructorShouldThrowWhenModelIdIsNull()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new VoyageAITextEmbeddingGenerationService(
                modelId: null!,
                apiKey: "test-api-key"
            )
        );
    }

    [Fact]
    public void ConstructorShouldThrowWhenApiKeyIsNull()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new VoyageAITextEmbeddingGenerationService(
                modelId: "voyage-3-large",
                apiKey: null!
            )
        );
    }

    [Fact]
    public async Task GenerateEmbeddingsAsyncShouldReturnEmbeddings()
    {
        // Arrange
        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = new[] { 0.1f, 0.2f, 0.3f }, index = 0, @object = "embedding" },
                new { embedding = new[] { 0.4f, 0.5f, 0.6f }, index = 1, @object = "embedding" }
            },
            usage = new { total_tokens = 10 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };

        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-3-large",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var data = new List<string> { "text1", "text2" };

        // Act
        var result = await service.GenerateEmbeddingsAsync(data).ConfigureAwait(false);

        // Assert
        result.Should().NotBeNull();
        result.Should().HaveCount(2);
        result[0].Length.Should().Be(3);
        result[1].Length.Should().Be(3);
    }

    [Fact]
    public async Task GenerateEmbeddingsAsyncShouldSendCorrectRequest()
    {
        // Arrange
        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = new[] { 0.1f, 0.2f }, index = 0, @object = "embedding" }
            },
            usage = new { total_tokens = 5 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };

        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-3-large",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var data = new List<string> { "test text" };

        // Act
        await service.GenerateEmbeddingsAsync(data).ConfigureAwait(false);

        // Assert
        this._messageHandlerStub.RequestContent.Should().NotBeNull();
        this._messageHandlerStub.RequestContent.Should().Contain("voyage-3-large");
        this._messageHandlerStub.RequestContent.Should().Contain("test text");
        this._messageHandlerStub.RequestHeaders.Should().ContainKey("Authorization");
    }

    [Fact]
    public async Task GenerateEmbeddingsAsyncShouldApplyExecutionSettings()
    {
        // Arrange
        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = new[] { 0.1f, 0.2f }, index = 0, @object = "embedding" }
            },
            usage = new { total_tokens = 5 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };

        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-3-large",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var settings = new VoyageAIEmbeddingPromptExecutionSettings
        {
            InputType = "document",
            Truncation = false,
            OutputDimension = 512,
            OutputDtype = "int8"
        };

        // Act
        await service.GenerateEmbeddingsAsync(new List<string> { "test text" }, settings).ConfigureAwait(false);

        // Assert
        using var doc = JsonDocument.Parse(this._messageHandlerStub.RequestContent!);
        var root = doc.RootElement;
        root.GetProperty("input_type").GetString().Should().Be("document");
        root.GetProperty("truncation").GetBoolean().Should().BeFalse();
        root.GetProperty("output_dimension").GetInt32().Should().Be(512);
        root.GetProperty("output_dtype").GetString().Should().Be("int8");
    }

    [Fact]
    public async Task GenerateEmbeddingsAsyncShouldHandleApiError()
    {
        // Arrange
        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.BadRequest)
        {
            Content = new StringContent("API error")
        };

        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-3-large",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var data = new List<string> { "test" };

        // Act & Assert
        await Assert.ThrowsAsync<HttpRequestException>(
            async () => await service.GenerateEmbeddingsAsync(data).ConfigureAwait(false)
        ).ConfigureAwait(false);
    }

    [Fact]
    public void ServiceShouldUseCustomEndpoint()
    {
        // Arrange
        var customEndpoint = "https://custom.api.com/v1";

        // Act
        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-4-large",
            apiKey: "test-api-key",
            endpoint: customEndpoint,
            httpClient: this._httpClient
        );

        // Assert
        service.Should().NotBeNull();
    }

    [Fact]
    public async Task ServiceShouldRouteMongoDBApiKeyToMongoDBEndpointAsync()
    {
        // Arrange: an "al-" prefixed key must be routed to the MongoDB base URL,
        // mirroring the official voyageai package (get_default_base_url).
        this.SetSuccessResponse();

        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-4-large",
            apiKey: "al-test-api-key",
            httpClient: this._httpClient
        );

        // Act
        await service.GenerateEmbeddingsAsync(new List<string> { "text" }).ConfigureAwait(false);

        // Assert
        this._messageHandlerStub.RequestUri.Should().NotBeNull();
        this._messageHandlerStub.RequestUri!.ToString().Should().StartWith("https://ai.mongodb.com/v1");
    }

    [Fact]
    public async Task ServiceShouldRouteVoyageAiApiKeyToVoyageAiEndpointAsync()
    {
        // Arrange: a non-"al-" key must be routed to the VoyageAI base URL.
        this.SetSuccessResponse();

        var service = new VoyageAITextEmbeddingGenerationService(
            modelId: "voyage-4-large",
            apiKey: "pa-test-api-key",
            httpClient: this._httpClient
        );

        // Act
        await service.GenerateEmbeddingsAsync(new List<string> { "text" }).ConfigureAwait(false);

        // Assert
        this._messageHandlerStub.RequestUri.Should().NotBeNull();
        this._messageHandlerStub.RequestUri!.ToString().Should().StartWith("https://api.voyageai.com/v1");
    }

    private void SetSuccessResponse()
    {
        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = new[] { 0.1f, 0.2f }, index = 0, @object = "embedding" }
            },
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
                this.RequestContent = await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            }

            foreach (var header in request.Headers)
            {
                this.RequestHeaders[header.Key] = string.Join(",", header.Value);
            }

            return this.ResponseToReturn ?? new HttpResponseMessage(HttpStatusCode.OK);
        }
    }
}
