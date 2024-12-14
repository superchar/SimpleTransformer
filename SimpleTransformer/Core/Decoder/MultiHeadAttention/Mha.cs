namespace Core.Decoder.MultiHeadAttention;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


public class Mha : Module<Tensor, Tensor>
{
    private readonly List<Head> _heads;
    
    public Mha(string name, int hiddenSize, int numHeads) : base(name)
    {
        var headDim = hiddenSize / numHeads;
        _heads = Enumerable.Range(0, numHeads)
            .Select(n => new Head($"Head {n}", headDim, hiddenSize))
            .ToList();
    }
    
    public override Tensor forward(Tensor input)
    {
        var results = _heads.Select(h => h.forward(input)).ToList();

        return cat(results, 2);
    }
}