// Copyright (c) Microsoft. All rights reserved.

using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI.Core;

/// <summary>
/// HTTP client for the VoyageAI by MongoDB API.
/// </summary>
internal sealed class VoyageAIClient
{
    /// <summary>Default base URL for VoyageAI API keys.</summary>
    internal const string VoyageAIBaseUrl = "https://api.voyageai.com/v1";

    /// <summary>Base URL used for MongoDB-issued API keys (prefixed with "al-").</summary>
    internal const string MongoDBBaseUrl = "https://ai.mongodb.com/v1";

    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private readonly string _endpoint;
    private readonly ILogger _logger;

    /// <summary>
    /// Returns the default base URL for the given API key, mirroring the official
    /// voyageai Python package (get_default_base_url): keys starting with "al-" are
    /// routed to the MongoDB endpoint, all others to the VoyageAI endpoint.
    /// </summary>
    /// <param name="apiKey">The VoyageAI by MongoDB API key.</param>
    /// <returns>The base URL to use for API requests.</returns>
    internal static string GetDefaultBaseUrl(string apiKey)
        => apiKey.StartsWith("al-", StringComparison.Ordinal) ? MongoDBBaseUrl : VoyageAIBaseUrl;

    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) }
    };

    public VoyageAIClient(
        string apiKey,
        string? endpoint = null,
        HttpClient? httpClient = null,
        ILogger? logger = null)
    {
        this._apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        this._endpoint = endpoint ?? GetDefaultBaseUrl(apiKey);
        this._httpClient = httpClient ?? new HttpClient();
        this._logger = logger ?? NullLogger.Instance;
    }

    public async Task<T> SendRequestAsync<T>(
        string path,
        object requestBody,
        CancellationToken cancellationToken = default)
    {
        var requestUri = $"{this._endpoint}/{path}";

        using var request = new HttpRequestMessage(HttpMethod.Post, requestUri);
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", this._apiKey);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

        var json = JsonSerializer.Serialize(requestBody, s_jsonOptions);
        request.Content = new StringContent(json, Encoding.UTF8, "application/json");

        this._logger.LogDebug("Sending VoyageAI request to {Uri}", requestUri);

        using var response = await this._httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);

        var responseContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            this._logger.LogError("VoyageAI API request failed with status {StatusCode}: {Response}",
                response.StatusCode, responseContent);
            throw new HttpRequestException(
                $"VoyageAI API request failed with status {response.StatusCode}: {responseContent}");
        }

        var result = JsonSerializer.Deserialize<T>(responseContent, s_jsonOptions);
        if (result is null)
        {
            throw new JsonException($"Failed to deserialize VoyageAI response: {responseContent}");
        }

        return result;
    }
}
