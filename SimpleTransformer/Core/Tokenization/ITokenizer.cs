namespace Core.Tokenization;

public interface ITokenizer
{
    int VocabSize { get; }
    
    int[] Encode(string text);

    (int[,] Tokens, PaddingMask Mask) EncodeMultiple(string[] texts);

    string Decode(int[] tokens);
}