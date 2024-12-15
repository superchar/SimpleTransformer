
using System.Text.RegularExpressions;
using Core.Decoder;
using Core.Tokenization;

const string tokenizerContent =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
const int vocabSize = 400;
var pattern = new Regex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+\n");
var numHeads = 2;
var hiddenSize = 32;
var contextSize = 20;
var blocksCount = 2;

var mergeableRanks = BpeTokenizer.Train(tokenizerContent, vocabSize, pattern);
var tokenizer = new BpeTokenizer(mergeableRanks, pattern);
var decoder = new TransformerDecoder(numHeads, hiddenSize, contextSize, blocksCount, tokenizer);
string[] prompts = ["Lorem", " ipsum dolor"];
var res = decoder.CompleteSeq(prompts, 10);
var test = "test";
