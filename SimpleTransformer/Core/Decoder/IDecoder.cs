namespace Core.Decoder;

public interface IDecoder
{
    string[] CompleteSeq(string[] prompts, int tokensCount);

    float Train(string text, int iterations, int batchSize);
}