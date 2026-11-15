// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Xunit;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI.UnitTests.Services;

/// <summary>
/// Unit tests for <see cref="VoyageAIMultimodalEmbeddingGenerationService"/>.
/// </summary>
public sealed class VoyageAIMultimodalEmbeddingGenerationServiceTests : IDisposable
{
    private readonly HttpMessageHandlerStub _messageHandlerStub;
    private readonly HttpClient _httpClient;

    public VoyageAIMultimodalEmbeddingGenerationServiceTests()
    {
        this._messageHandlerStub = new HttpMessageHandlerStub();
        this._httpClient = new HttpClient(this._messageHandlerStub);
    }

    [Fact]
    public void Constructor_WithValidParameters_CreatesInstance()
    {
        // Arrange & Act
        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        // Assert
        service.Should().NotBeNull();
        service.Attributes.Should().ContainKey("ModelId");
        service.Attributes["ModelId"].Should().Be("voyage-multimodal-3");
    }

    [Fact]
    public void Constructor_WithVoyageMultimodal35_CreatesInstance()
    {
        // Arrange & Act
        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3.5",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        // Assert
        service.Should().NotBeNull();
        service.Attributes.Should().ContainKey("ModelId");
        service.Attributes["ModelId"].Should().Be("voyage-multimodal-3.5");
    }

    [Fact]
    public void Constructor_WithNullModelId_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new VoyageAIMultimodalEmbeddingGenerationService(
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
            new VoyageAIMultimodalEmbeddingGenerationService(
                modelId: "voyage-multimodal-3",
                apiKey: null!
            )
        );
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsWithTextInputsReturnsEmbeddingsAsync()
    {
        // Arrange
        var expectedEmbeddings = new List<float[]>
        {
            new[] { 0.1f, 0.2f, 0.3f },
            new[] { 0.4f, 0.5f, 0.6f }
        };

        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = expectedEmbeddings[0], index = 0, @object = "embedding" },
                new { embedding = expectedEmbeddings[1], index = 1, @object = "embedding" }
            },
            usage = new { total_tokens = 20 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var inputs = new List<VoyageAIMultimodalInput>
        {
            VoyageAIMultimodalInput.FromText("text1"),
            VoyageAIMultimodalInput.FromText("text2")
        };

        // Act
        var result = await service.GenerateMultimodalEmbeddingsAsync(inputs);

        // Assert
        result.Should().NotBeNull();
        result.Should().HaveCount(2);
        result[0].Length.Should().Be(3);
        result[1].Length.Should().Be(3);
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsWithMixedInputsReturnsEmbeddingsAsync()
    {
        // Arrange
        var expectedEmbeddings = new List<float[]>
        {
            new[] { 0.1f, 0.2f, 0.3f },
            new[] { 0.4f, 0.5f, 0.6f }
        };

        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = expectedEmbeddings[0], index = 0, @object = "embedding" },
                new { embedding = expectedEmbeddings[1], index = 1, @object = "embedding" }
            },
            usage = new { total_tokens = 30 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        // Interleaved text + image within a single input, plus a text-only input.
        var inputs = new List<VoyageAIMultimodalInput>
        {
            new(
                VoyageAIMultimodalContent.FromText("A banana"),
                VoyageAIMultimodalContent.FromImageUrl("https://example.com/banana.jpg")),
            VoyageAIMultimodalInput.FromText("text2")
        };

        // Act
        var result = await service.GenerateMultimodalEmbeddingsAsync(inputs);

        // Assert
        result.Should().NotBeNull();
        result.Should().HaveCount(2);
        result[0].Length.Should().Be(3);
        result[1].Length.Should().Be(3);
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsSendsStructuredContentPayloadAsync()
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

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var inputs = new List<VoyageAIMultimodalInput>
        {
            new(
                VoyageAIMultimodalContent.FromText("a caption"),
                VoyageAIMultimodalContent.FromImageBase64("data:image/png;base64,iVBORw0KGgo="))
        };

        // Act
        await service.GenerateMultimodalEmbeddingsAsync(inputs);

        // Assert - verify the official structured payload shape
        this._messageHandlerStub.RequestContent.Should().NotBeNull();
        using var doc = JsonDocument.Parse(this._messageHandlerStub.RequestContent!);
        var root = doc.RootElement;
        root.GetProperty("model").GetString().Should().Be("voyage-multimodal-3");

        var inputsElement = root.GetProperty("inputs");
        inputsElement.GetArrayLength().Should().Be(1);

        var content = inputsElement[0].GetProperty("content");
        content.GetArrayLength().Should().Be(2);

        content[0].GetProperty("type").GetString().Should().Be("text");
        content[0].GetProperty("text").GetString().Should().Be("a caption");

        content[1].GetProperty("type").GetString().Should().Be("image_base64");
        content[1].GetProperty("image_base64").GetString().Should().Be("data:image/png;base64,iVBORw0KGgo=");
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsAppliesExecutionSettingsAsync()
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

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var settings = new VoyageAIMultimodalEmbeddingPromptExecutionSettings
        {
            InputType = "query",
            Truncation = false
        };

        // Act
        await service.GenerateEmbeddingsAsync(new List<string> { "hello" }, settings);

        // Assert
        using var doc = JsonDocument.Parse(this._messageHandlerStub.RequestContent!);
        var root = doc.RootElement;
        root.GetProperty("input_type").GetString().Should().Be("query");
        root.GetProperty("truncation").GetBoolean().Should().BeFalse();
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsSendsCorrectRequestAsync()
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

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var inputs = new List<VoyageAIMultimodalInput> { VoyageAIMultimodalInput.FromText("test text") };

        // Act
        await service.GenerateMultimodalEmbeddingsAsync(inputs);

        // Assert
        this._messageHandlerStub.RequestContent.Should().NotBeNull();
        this._messageHandlerStub.RequestContent.Should().Contain("voyage-multimodal-3");
        this._messageHandlerStub.RequestContent.Should().Contain("test text");
        this._messageHandlerStub.RequestHeaders.Should().ContainKey("Authorization");
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsSendsCorrectModelAsync()
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

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var inputs = new List<VoyageAIMultimodalInput> { VoyageAIMultimodalInput.FromText("test text") };

        // Act
        var result = await service.GenerateMultimodalEmbeddingsAsync(inputs);

        // Assert
        result.Should().NotBeNull();
        this._messageHandlerStub.RequestContent.Should().NotBeNull();
        this._messageHandlerStub.RequestContent.Should().Contain("voyage-multimodal-3");
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsHandlesApiErrorAsync()
    {
        // Arrange
        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.InternalServerError)
        {
            Content = new StringContent("API error")
        };

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var inputs = new List<VoyageAIMultimodalInput> { VoyageAIMultimodalInput.FromText("test") };

        // Act & Assert
        await Assert.ThrowsAsync<HttpRequestException>(
            async () => await service.GenerateMultimodalEmbeddingsAsync(inputs));
    }

    [Fact]
    public async Task GenerateEmbeddingsWrapsToMultimodalCallAsync()
    {
        // Arrange
        var expectedEmbedding = new[] { 0.1f, 0.2f, 0.3f };

        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = expectedEmbedding, index = 0, @object = "embedding" }
            },
            usage = new { total_tokens = 10 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var data = new List<string> { "text1" };

        // Act
        var result = await service.GenerateEmbeddingsAsync(data);

        // Assert
        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        result[0].Length.Should().Be(3);

        // The text path must produce the structured single-item content payload.
        using var doc = JsonDocument.Parse(this._messageHandlerStub.RequestContent!);
        var content = doc.RootElement.GetProperty("inputs")[0].GetProperty("content");
        content[0].GetProperty("type").GetString().Should().Be("text");
        content[0].GetProperty("text").GetString().Should().Be("text1");
    }

    [Fact]
    public void Attributes_ContainsModelId()
    {
        // Arrange & Act
        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        // Assert
        service.Attributes.Should().ContainKey("ModelId");
        service.Attributes["ModelId"].Should().Be("voyage-multimodal-3");
    }

    [Fact]
    public void ServiceShouldUseCustomEndpoint()
    {
        // Arrange
        var customEndpoint = "https://custom.api.com/v1";

        // Act
        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3",
            apiKey: "test-api-key",
            endpoint: customEndpoint,
            httpClient: this._httpClient
        );

        // Assert
        service.Should().NotBeNull();
    }

    [Fact]
    public async Task GenerateMultimodalEmbeddingsWithVoyageMultimodal35SendsCorrectModelAsync()
    {
        // Arrange
        var responseContent = JsonSerializer.Serialize(new
        {
            data = new[]
            {
                new { embedding = new[] { 0.1f, 0.2f, 0.3f, 0.4f }, index = 0, @object = "embedding" }
            },
            usage = new { total_tokens = 5 }
        });

        this._messageHandlerStub.ResponseToReturn = new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(responseContent)
        };

        var service = new VoyageAIMultimodalEmbeddingGenerationService(
            modelId: "voyage-multimodal-3.5",
            apiKey: "test-api-key",
            httpClient: this._httpClient
        );

        var inputs = new List<VoyageAIMultimodalInput> { VoyageAIMultimodalInput.FromText("test text for multimodal 3.5") };

        // Act
        var result = await service.GenerateMultimodalEmbeddingsAsync(inputs);

        // Assert
        result.Should().NotBeNull();
        result.Should().HaveCount(1);
        this._messageHandlerStub.RequestContent.Should().NotBeNull();
        this._messageHandlerStub.RequestContent.Should().Contain("voyage-multimodal-3.5");
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
        public Dictionary<string, string> RequestHeaders { get; } = new();

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
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
