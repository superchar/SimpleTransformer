
using System.Text.RegularExpressions;
using Core.Decoder;
using Core.Tokenization;

const string trainingContent =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
const int vocabSize = 400;
var pattern = new Regex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+\n");
const int numHeads = 2;
const int hiddenSize = 32;
const int contextSize = 10;
const int blocksCount = 2;
const int trainingIterations = 100;
const int trainingBatchSize = 5;

var mergeableRanks = BpeTokenizer.Train(trainingContent, vocabSize, pattern);
var tokenizer = new BpeTokenizer(mergeableRanks, pattern);
var decoder = new TransformerDecoder(numHeads, hiddenSize, contextSize, blocksCount, tokenizer);
string[] prompts = ["Lorem", " ipsum dolor"];
decoder.Train(trainingContent, trainingIterations, trainingBatchSize);
var results = decoder.CompleteSeq(prompts, 5);

foreach (var result in results)
{
    Console.WriteLine(string.Join(',', result));
}
