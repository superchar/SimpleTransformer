using TorchSharp.Modules;

namespace Core.Decoder.MultiHeadAttention;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


public class Mha : Module<Tensor, Tensor>
{
    private readonly List<Head> _heads;
    private readonly Linear _projection;
    
    public Mha(string name, int hiddenSize, int numHeads) : base(name)
    {
        _heads = CreateHeads(numHeads, hiddenSize);
        _projection = Linear(hiddenSize, hiddenSize);
    }
    
    public override Tensor forward(Tensor input)
    {
        var results = _heads.Select(h => h.forward(input)).ToList();

        return _projection.forward(cat(results, 2));
    }
    
    private List<Head> CreateHeads(int numHeads, int hiddenSize)
    {
        var headDim = hiddenSize / numHeads;
        var heads = Enumerable.Range(0, numHeads)
            .Select(n => new Head($"Head {n}", headDim, hiddenSize))
            .ToList();
        
        foreach (var head in heads)
        {
            register_module(head.GetName(), head);
        }

        return heads;
    }
}