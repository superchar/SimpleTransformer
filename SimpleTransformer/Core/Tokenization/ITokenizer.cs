namespace Core.Tokenization;

public interface ITokenizer
{
    int VocabSize { get; }
    
    int[] Encode(string text);

    int[,] EncodeMultiple(string[] texts);

    string Decode(int[] tokens);
}