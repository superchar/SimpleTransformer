using System.Text.RegularExpressions;

namespace Core.Tokenization;

public interface ITokenizer
{
    void Train(string content, int vocabSize, Regex wordRegex);
    
    int VocabSize { get; }
    
    int[] Encode(string text);

    (int[,] Tokens, PaddingMask Mask) EncodeMultiple(string[] texts);

    string Decode(int[] tokens);
}