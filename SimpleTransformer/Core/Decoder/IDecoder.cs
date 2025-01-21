namespace Core.Decoder;

public interface IDecoder
{
    IEnumerable<string[]> CompleteSeq(string[] prompts);

    float Train(string text, int iterations, int batchSize);
}