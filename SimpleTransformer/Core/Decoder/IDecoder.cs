namespace Core.Decoder;

public interface IDecoder
{
    string[] CompleteSeq(string[] prompts, int tokensCount);
}