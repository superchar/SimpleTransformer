using Core.Tokenization;

namespace Core.Decoder.Factory;

public class TransformerDecoderFactory(ITokenizer tokenizer) : IDecoderFactory
{
    public IDecoder CreateDecoder(int numHeads, int hiddenSize, int contextSize, int blocksCount)
        => new TransformerDecoder(numHeads, hiddenSize, contextSize, blocksCount, tokenizer);
}