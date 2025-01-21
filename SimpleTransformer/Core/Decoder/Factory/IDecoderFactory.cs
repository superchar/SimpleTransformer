namespace Core.Decoder.Factory;

public interface IDecoderFactory
{
    IDecoder CreateDecoder(int numHeads, int hiddenSize, int contextSize, int blocksCount);
}