using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Core.Decoder.MultiHeadAttention;

public class Head : Module<Tensor, Tensor>
{
    private readonly Linear _key;
    private readonly Linear _query;
    private readonly Linear _value;
    
    public Head(string name, int headSize, int hiddenSize) : base(name)
    {
        _key = Linear(hiddenSize, headSize);
        _query = Linear(hiddenSize, headSize);
        _value = Linear(hiddenSize, headSize);
    }

    public override Tensor forward(Tensor input)
    {
        var keys = _key.forward(input);
        var queries = _query.forward(input);
        var values = _value.forward(input);
        var wei = keys.matmul(queries.transpose(2, 1)) / Math.Sqrt(keys.shape[1]); // SDPA, Essentially kv dot product divided by sqrt of head size and softmaxed. Very inefficient in terms of GPU memory, needs seq_size ^ 2
        var mask = tril(ones(input.shape[1], input.shape[1])).eq(0); // seq_length x seq_length
        wei = wei.masked_fill(mask, float.NegativeInfinity); // Because we already applied tril mask, padded tokens will be ignored for real tokens
        wei = softmax(wei, 2);

        return wei.matmul(values);
    }
}