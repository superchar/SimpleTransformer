using System.Text.RegularExpressions;
using CliFx;
using CliFx.Attributes;
using CliFx.Infrastructure;
using Core.Decoder.Factory;
using Core.Tokenization;

namespace CLI;

[Command("run-transformer")]
public class RunTransformerCommand(ITokenizer tokenizer, IDecoderFactory decoderFactory) : ICommand
{
    private const string DefaultTrainingContent =
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
    
    [CommandOption("num-heads", 'h', Description = "Number of heads in the transformer model.")]
    public int NumHeads { get; set; } = 2;
    
    [CommandOption("hidden-size", 's', Description = "Size of the hidden layer in the transformer model.")]
    public int HiddenSize { get; set; } = 32;
    
    [CommandOption("context-size", 'c', Description = "Size of the context in the transformer model.")]
    public int ContextSize { get; set; } = 10;
    
    [CommandOption("blocks-count", 'b', Description = "Number of blocks in the transformer model.")]
    public int BlocksCount { get; set; } = 2;
    
    [CommandOption("training-iterations", 'i', Description = "Number of training iterations.")]
    public int TrainingIterations { get; set; } = 100;
    
    [CommandOption("training-batch-size", 't', Description = "Size of the training batch.")]
    public int TrainingBatchSize { get; set; } = 5;

    [CommandOption("tokenizer-pattern", 'p', Description = "Pattern for the tokenizer.")]
    public string TokenizerPattern { get; set; } =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+\n";
    
    [CommandOption("vocab-size", 'v', Description = "Size of the vocabulary.")]
    public int VocabSize { get; set; } = 1000;
    
    [CommandOption("training-file-path", 'f', Description = "Path to the training content file.")]
    public string TrainingFilePath { get; set; }
    
    public async ValueTask ExecuteAsync(IConsole console)
    {
        var trainingContent = await GetTrainingContentAsync();
        await console.Output.WriteLineAsync("Training the tokenizer...");
        tokenizer.Train(trainingContent, VocabSize, new Regex(TokenizerPattern));
        await console.Output.WriteLineAsync("Tokenizer trained successfully.");
        await console.Output.WriteLineAsync("Training the decoder...");
        var decoder = decoderFactory.CreateDecoder(NumHeads, HiddenSize, ContextSize, BlocksCount);
        decoder.Train(trainingContent, TrainingIterations, TrainingBatchSize);
        await console.Output.WriteLineAsync("Decoder trained successfully.");
        while (true)
        {
            await console.Output.WriteLineAsync("Enter a prompt:");
            var prompt = await console.Input.ReadLineAsync();
            await console.Output.WriteLineAsync("Enter the number of tokens to generate:");
            var tokensCount = int.Parse(await console.Input.ReadLineAsync());
            var result = decoder.CompleteSeq([prompt]);

            foreach (var tokens in result.Take(tokensCount))
            {
                await console.Output.WriteAsync(tokens[0]);
            }
            await console.Output.WriteLineAsync();
        }
    }

    private Task<string> GetTrainingContentAsync()
        => TrainingFilePath is not null
            ? File.ReadAllTextAsync(TrainingFilePath)
            : Task.FromResult(DefaultTrainingContent);
}