using System.Security.Cryptography;

namespace Core.Tokenization;

public record Bytes(byte[] Content)
{
    public Bytes Merge(Bytes otherBytes)
        => new (Content.Concat(otherBytes.Content).ToArray());

    public virtual bool Equals(Bytes? other)
        => other != null && other.Content.SequenceEqual(Content);

    public override int GetHashCode()
        => Content.Aggregate(0, HashCode.Combine);
}