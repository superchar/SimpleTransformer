namespace Core.Tokenization;

public interface ITokenizer
{
    int[] Encode(string text);

    string Decode(int[] tokens);
}