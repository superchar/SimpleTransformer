using Core.Decoder.Blocks;
using Core.Tokenization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Core.Decoder;

public class TransformerDecoder : Module<Tensor, Tensor>, IDecoder
{
    private readonly Linear _linear;
    private readonly Sequential _blocks;
    private readonly Embedding _tokenEmbeddings;
    private readonly Embedding _positionEmbeddings;
    private readonly ITokenizer _tokenizer;

    public TransformerDecoder(
        int numHeads,
        int hiddenSize,
        int contextSize,
        int blocksCount,
        ITokenizer tokenizer) : base("transformer")
    {
        _linear = Linear(hiddenSize, tokenizer.VocabSize);
        _blocks = Sequential(Enumerable.Range(0, blocksCount)
            .Select(n => new Block($"Block{n}", hiddenSize, numHeads))
            .ToList());
        _tokenEmbeddings = Embedding(tokenizer.VocabSize, hiddenSize);
        _positionEmbeddings = Embedding(contextSize, hiddenSize);
        _tokenizer = tokenizer;
    }
    
    public string[] CompleteSeq(string[] prompts, int tokensCount)
    {
        var generatedTokens = 0;
        do
        {
            var tokens = tensor(_tokenizer.EncodeMultiple(prompts));
            var embeddings = _tokenEmbeddings.forward(tokens);
            var positionalEmbeddings = _positionEmbeddings.forward(arange(0, embeddings.shape[1]));
            embeddings += positionalEmbeddings;

            var probs = forward(embeddings);
            for (var i = 0; i < prompts.Length; i++)
            {
                var nextToken = multinomial(probs[i][^1], 1).ToInt32();
                prompts[i] += _tokenizer.Decode([nextToken]);
            }

            generatedTokens++;
        } while (generatedTokens < tokensCount);
        
        return prompts;
    }

    public override Tensor forward(Tensor input)
    {
        var linearResult = _linear.forward(_blocks.forward(input));

        return softmax(linearResult, 0);
    }
}