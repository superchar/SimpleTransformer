namespace Core.Tokenization;

public record PaddingMask(int[,] Array)
{
    public int GetLastNonPaddedIndex(int seqIndex)
    {
        var maxSeqLength = Array.GetUpperBound(1) + 1;
        for (var i = 0; i < maxSeqLength; i++)
        {
            if (Array[seqIndex, i] == 0)
            {
                return i - 1;
            }
        }

        return maxSeqLength - 1;
    }
}