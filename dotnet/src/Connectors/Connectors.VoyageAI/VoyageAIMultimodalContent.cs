// Copyright (c) Microsoft. All rights reserved.

using System.Diagnostics.CodeAnalysis;

namespace Microsoft.SemanticKernel.Connectors.VoyageAI;

/// <summary>
/// A single piece of content (text or image) that forms part of a multimodal input for
/// the VoyageAI by MongoDB multimodal embeddings API.
/// </summary>
/// <remarks>
/// Mirrors the content items documented at
/// https://docs.voyageai.com/reference/multimodal-embeddings-api. A single input is composed
/// of an ordered list of these items (see <see cref="VoyageAIMultimodalInput"/>).
/// </remarks>
[Experimental("SKEXP0001")]
public sealed class VoyageAIMultimodalContent
{
    /// <summary>The content type: one of "text", "image_url", or "image_base64".</summary>
    public string Type { get; }

    /// <summary>The text value, when <see cref="Type"/> is "text".</summary>
    public string? Text { get; }

    /// <summary>The image URL, when <see cref="Type"/> is "image_url".</summary>
    public string? ImageUrl { get; }

    /// <summary>The Base64 data URL (e.g. "data:image/png;base64,..."), when <see cref="Type"/> is "image_base64".</summary>
    public string? ImageBase64 { get; }

    private VoyageAIMultimodalContent(string type, string? text = null, string? imageUrl = null, string? imageBase64 = null)
    {
        this.Type = type;
        this.Text = text;
        this.ImageUrl = imageUrl;
        this.ImageBase64 = imageBase64;
    }

    /// <summary>Creates a text content item.</summary>
    /// <param name="text">The text to embed.</param>
    public static VoyageAIMultimodalContent FromText(string text)
    {
        Verify.NotNull(text);
        return new VoyageAIMultimodalContent("text", text: text);
    }

    /// <summary>Creates an image content item that references an image by URL.</summary>
    /// <param name="imageUrl">A URL linking to the image.</param>
    public static VoyageAIMultimodalContent FromImageUrl(string imageUrl)
    {
        Verify.NotNullOrWhiteSpace(imageUrl);
        return new VoyageAIMultimodalContent("image_url", imageUrl: imageUrl);
    }

    /// <summary>Creates an image content item from a Base64 data URL.</summary>
    /// <param name="imageBase64">A Base64 data URL, e.g. "data:image/png;base64,iVBORw0KG...".</param>
    public static VoyageAIMultimodalContent FromImageBase64(string imageBase64)
    {
        Verify.NotNullOrWhiteSpace(imageBase64);
        return new VoyageAIMultimodalContent("image_base64", imageBase64: imageBase64);
    }
}

/// <summary>
/// A single multimodal input for the VoyageAI by MongoDB multimodal embeddings API, composed of
/// one or more ordered <see cref="VoyageAIMultimodalContent"/> items (interleaved text and images).
/// Each input produces one embedding.
/// </summary>
[Experimental("SKEXP0001")]
public sealed class VoyageAIMultimodalInput
{
    /// <summary>The ordered content items (text and/or images) that make up this input.</summary>
    public IList<VoyageAIMultimodalContent> Content { get; }

    /// <summary>Initializes a new instance of the <see cref="VoyageAIMultimodalInput"/> class.</summary>
    /// <param name="content">The ordered content items for this input.</param>
    public VoyageAIMultimodalInput(IList<VoyageAIMultimodalContent> content)
    {
        Verify.NotNull(content);
        this.Content = content;
    }

    /// <summary>Initializes a new instance of the <see cref="VoyageAIMultimodalInput"/> class.</summary>
    /// <param name="content">The ordered content items for this input.</param>
    public VoyageAIMultimodalInput(params VoyageAIMultimodalContent[] content)
        : this((IList<VoyageAIMultimodalContent>)content)
    {
    }

    /// <summary>Creates an input consisting of a single text item.</summary>
    /// <param name="text">The text to embed.</param>
    public static VoyageAIMultimodalInput FromText(string text)
        => new(VoyageAIMultimodalContent.FromText(text));
}
