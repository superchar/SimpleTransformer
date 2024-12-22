using Core.Decoder.FeedForward;
using Core.Decoder.MultiHeadAttention;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using LayerNorm = TorchSharp.Modules.LayerNorm;

namespace Core.Decoder.Blocks;

public class Block : Module<Tensor, Tensor>
{
    private readonly Mha _mha;
    private readonly Fnn _fnn;
    private readonly LayerNorm _mhaLayerNorm;
    private readonly LayerNorm _fnnLayerNorm;
    
    public Block(string name, int hiddenSize, int numHeads) : base(name)
    {
        _mha = new Mha("mha", hiddenSize, numHeads);
        _fnn = new Fnn("fnn", hiddenSize);
        _mhaLayerNorm = LayerNorm(hiddenSize);
        _fnnLayerNorm = LayerNorm(hiddenSize);
    }

    public override Tensor forward(Tensor input)
    {
        var res = _mhaLayerNorm.forward(_mha.forward(input) + input); // Normalize to 0 mean and 1 deviation
        return _fnnLayerNorm.forward(_fnn.forward(input) + res); // Applying normalization after residual connection
    }
}