using TorchSharp;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using Linear = TorchSharp.Modules.Linear;

namespace Core.Decoder.MultiHeadAttention;

public class Head : Module<Tensor, Tensor>
{
    private Linear _key;
    private Linear _query;
    private Linear _value;
    
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
        wei = softmax(wei, 0);

        return wei.matmul(values);
    }
}