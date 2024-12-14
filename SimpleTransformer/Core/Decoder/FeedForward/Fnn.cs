using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Core.Decoder.FeedForward;

public class Fnn : Module<Tensor, Tensor>
{
    private readonly Sequential _sequential;
    
    public Fnn(string name, int hiddenSize) : base(name)
    {
        var firstLin = Linear(hiddenSize, hiddenSize * 4);
        var relu = ReLU();
        var lastLin = Linear(hiddenSize * 4, hiddenSize);
        _sequential = Sequential(("firstLin", firstLin), ("relu", relu), ("lastLin", lastLin));
    }

    public override Tensor forward(Tensor input)
        => _sequential.forward(input);
}