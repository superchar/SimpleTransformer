using System.Text.RegularExpressions;
using Core.Tokenization;
using FluentAssertions;

namespace Core.Tests;

public class BpeTokenizerTests
{
    private const int VocabSize = 1000;
    private const string TrainContent =
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum vestibulum. Cras venenatis euismod malesuada.";
    
    private static readonly Regex WordRegex =
        new("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+\n");
    
    private BpeTokenizer _bpeTokenizer;

    [SetUp]
    public void Setup()
    {
        var mergeableRanks = BpeTokenizer.Train(TrainContent, VocabSize, WordRegex);

        _bpeTokenizer = new BpeTokenizer(mergeableRanks, WordRegex);
    }
    
    [TestCase("Lorem ipsum dolor sit amet", 5)]
    [TestCase(" Vivamus lacinia odio vitae", 4)]
    [TestCase(" malesuada", 1)]
    [TestCase("", 0)]
    [TestCase(null, 0)]
    public void Encode_ReturnsCorrectTokensCount(string content, int expectedTokensCount)
    {
        var result = _bpeTokenizer.Encode(content);

        result.Should().HaveCount(expectedTokensCount);
    }
    
    [TestCase("Lorem ipsum dolor sit amet")]
    [TestCase(" Vivamus lacinia odio vitae")]
    [TestCase(" malesuada")]
    [TestCase("")]
    [TestCase("Totally new words")]
    public void Decode_ReturnsOriginalStringAfterEncoding(string content)
    {
        var encodedString = _bpeTokenizer.Encode(content);
        
        var result = _bpeTokenizer.Decode(encodedString);

        result.Should().BeEquivalentTo(content);
    }
    
    [Test]
    public void Decode_NullInput_ReturnsEmptyString()
    {
        var result = _bpeTokenizer.Decode(null);

        result.Should().BeEmpty();
    }
    
    [TestCase(0x80)]
    [TestCase(0xBF)]
    [TestCase(0x9A)]
    public void Decode_InvalidUtf8Bytes_DoesNotThrowException(byte b)
    {
        _bpeTokenizer.Invoking(t => t.Decode([b])).Should().NotThrow();
    }

    [Test]
    public void Train_VocabSizeIsBiggerThanTotalPairs_ReturnsSmallerVocabSize()
    {
        var result = BpeTokenizer.Train(TrainContent, VocabSize, WordRegex);

        result.Count.Should().BeLessThan(VocabSize);
    }
    
    [Test]
    public void Train_VocabSizeIsSmallerThanTotalPairs_ReturnsSpecifiedVocabSize()
    {
        const int vocabSize = byte.MaxValue + 1;
        var result = BpeTokenizer.Train(TrainContent, vocabSize, WordRegex);

        result.Count.Should().Be(vocabSize);
    }
}