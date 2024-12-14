using System.Text;
using System.Text.RegularExpressions;

namespace Core.Tokenization;

public class BpeTokenizer : ITokenizer
{
    private const int PaddingToken = 0;
    
    private static readonly UTF8Encoding Utf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: false);

    private readonly Regex _wordRegex;
    private readonly Dictionary<int, Bytes> _decoder;
    private readonly Dictionary<Bytes, int> _mergeableRanks;

    public BpeTokenizer(Dictionary<Bytes, int> mergeableRanks, Regex wordRegex)
    {
        _mergeableRanks = mergeableRanks;
        _decoder = mergeableRanks
            .ToDictionary(kv => kv.Value, kv => kv.Key);
        _wordRegex = wordRegex;
    }

    public int VocabSize => _mergeableRanks.Count;

    public static Dictionary<Bytes, int> Train(string content, int vocabSize, Regex wordRegex)
    {
        var mergeableRanks = new Dictionary<Bytes, int>();
        for (byte i = 0; i < byte.MaxValue; i++)
        {
            mergeableRanks[new Bytes([i])] = i;
        }

        mergeableRanks[new Bytes([byte.MaxValue])] = byte.MaxValue;

        var words = wordRegex.Matches(content)
            .Select(m => Utf8.GetBytes(m.Value)
                .Select(b => new Bytes([b]))
                .ToList())
            .ToList();

        while (mergeableRanks.Count < vocabSize)
        {
            var stats = new Dictionary<(Bytes, Bytes), int>();
            foreach (var word in words)
            {
                for (var i = 0; i < word.Count - 1; i++)
                {
                    var pair = (word[i], word[i + 1]);
                    stats[pair] = stats.GetValueOrDefault(pair, 0) + 1;
                }
            }

            if (stats.Count == 0)
            {
                break;
            }

            var (pairToMerge, _) = stats.MaxBy(kv => kv.Value);
            var mergedPair = pairToMerge.Item1.Merge(pairToMerge.Item2);
            mergeableRanks[mergedPair] = mergeableRanks.Count;

            for (var i = 0; i < words.Count; i++)
            {
                var word = words[i];
                var newWord = new List<Bytes>();
                for (var j = 0; j < word.Count; j++)
                {
                    if (j < word.Count - 1 && (word[j], word[j + 1]).Equals(pairToMerge))
                    {
                        newWord.Add(mergedPair);
                        j++;
                    }
                    else
                    {
                        newWord.Add(word[j]);
                    }
                }

                words[i] = newWord;
            }
        }

        return mergeableRanks;
    }

    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return [];
        }

        var words = _wordRegex.Matches(text)
            .Select(m => Encoding.UTF8.GetBytes(m.Value)
                .Select(b => new Bytes([b]))
                .ToList())
            .ToList();

        var result = new List<int>();
        foreach (var word in words)
        {
            int minIndex;
            do
            {
                var minRank = int.MaxValue;
                minIndex = -1;
                for (var j = 0; j < word.Count; j++)
                {
                    if (j < word.Count - 1)
                    {
                        var mergedPair = word[j].Merge(word[j + 1]);
                        if (_mergeableRanks.TryGetValue(mergedPair, out var rank) && rank < minRank)
                        {
                            minIndex = j;
                        }
                    }
                }

                if (minIndex != -1)
                {
                    word[minIndex] = word[minIndex].Merge(word[minIndex + 1]);
                    word.RemoveAt(minIndex + 1);
                }
            } while (minIndex != -1);

            result.AddRange(word.Select(b => _mergeableRanks[b]));
        }

        return result
            .ToArray();
    }

    public int[,] EncodeMultiple(string[] texts)
    {
        var tokens = texts.Select(Encode).ToList();
        var largestInput = tokens.Max(t => t.Length);
        var result = new int[texts.Length, largestInput];
        for (var i = 0; i < texts.Length; i++)
        {
            var promptTokens = tokens[i];
            for (var j = 0; j < largestInput; j++)
            {
                var token = j < promptTokens.Length ? promptTokens[j] : PaddingToken;
                result[i, j] = token;
            }
        }

        return result;
    }


    public string Decode(int[] tokens)
        => tokens == null
            ? string.Empty
            : Utf8.GetString(tokens
                .Select(t => _decoder[t].Content)
                .SelectMany(c => c)
                .ToArray());
}