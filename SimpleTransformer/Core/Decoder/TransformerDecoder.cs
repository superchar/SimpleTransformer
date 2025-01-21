using Core.Decoder.Blocks;
using Core.Tokenization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Core.Decoder;

public class TransformerDecoder : Module<Tensor, Tensor>, IDecoder
{
    private readonly int _contextSize;
    
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
        _contextSize = contextSize;
        
        _linear = Linear(hiddenSize, tokenizer.VocabSize);
        _blocks = Sequential(Enumerable.Range(0, blocksCount)
            .Select(n => new Block($"Block{n}", hiddenSize, numHeads))
            .ToList());
        _tokenEmbeddings = Embedding(tokenizer.VocabSize, hiddenSize);
        _positionEmbeddings = Embedding(contextSize, hiddenSize);
        _tokenizer = tokenizer;
        RegisterComponents();
    }
    
    public IEnumerable<string[]> CompleteSeq(string[] prompts)
    {
        while (true)
        {
            var (tokens, paddingMask) = TrimToContextSize(_tokenizer.EncodeMultiple(prompts));
            var logits = forward(tensor(tokens)); // batch_size x seq_length x vocab_size
            var probs = softmax(logits, 2);
            var iterationTokens = new string[prompts.Length];
            for (var i = 0; i < prompts.Length; i++)
            {
                var lastRealToken = paddingMask.GetLastNonPaddedIndex(i);
                var nextToken = multinomial(probs[i][lastRealToken], 1).ToInt32();
                var nextDecodedToken = _tokenizer.Decode([nextToken]);
                iterationTokens[i] = nextDecodedToken;
                prompts[i] += nextDecodedToken;
            }

            yield return iterationTokens;
        }
    }

    public float Train(string text, int iterations, int batchSize)
    {
        var optimizer = optim.Adam(parameters(), lr: 0.01);
        var tokens = _tokenizer.Encode(text);
        float lastLoss = 0;
        for (var i = 0; i < iterations; i++)
        {
            var (x, y) = GetBatch(tokens, batchSize);
            var logits = forward(x).view(batchSize * _contextSize, _tokenizer.VocabSize);
            y = y.view(batchSize * _contextSize);
            var loss = functional.cross_entropy(logits, y);
            lastLoss = loss.item<float>();
            Console.WriteLine($"Last loss: {lastLoss}");
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        return lastLoss;
    }

    public override Tensor forward(Tensor tokens)
    {
        var embeddings = _tokenEmbeddings.forward(tokens);
        var positionalEmbeddings = _positionEmbeddings.forward(arange(0, embeddings.shape[1]));
        embeddings += positionalEmbeddings;
        return _linear.forward(_blocks.forward(embeddings));
    }

    private (int[,], PaddingMask) TrimToContextSize((int[,] tokens, PaddingMask paddingMask) data)
    {
        var (tokens, paddingMask) = data;
        var totalColumns = tokens.GetLength(1);

        if (totalColumns <= _contextSize)
        {
            return (tokens, paddingMask);
        }

        var result = new int[tokens.GetLength(0), _contextSize];
        var startColumn = totalColumns - _contextSize;

        for (var i = 0; i < tokens.GetLength(0); i++)
        {
            for (var j = 0; j < _contextSize; j++)
            {
                result[i, j] = tokens[i, startColumn + j];
            }
        }

        return (result, new PaddingMask(result));
    }
    
    private (Tensor X, Tensor Y) GetBatch(int[] data, int batchSize)
    {
        var rand = new Random();
        var offsets = Enumerable.Range(0, batchSize)
            .Select(_ => rand.Next(0, data.Length - (_contextSize + 1)))
            .ToList();

        var x = new int[batchSize, _contextSize];
        var y = new int[batchSize, _contextSize];
        for (var i = 0; i < batchSize; i++)
        {
            var xBatch = data[offsets[i]..(offsets[i] + _contextSize)];
            var yBatch = data[(offsets[i] + 1)..(offsets[i] + _contextSize + 1)];
            for (var j = 0; j < _contextSize; j++)
            {
                x[i, j] = xBatch[j];
                y[i, j] = yBatch[j];
            }
        }

        return (tensor(x), tensor(y, ScalarType.Int64));
    }
}