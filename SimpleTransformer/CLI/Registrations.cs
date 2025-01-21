using Core.Decoder.Factory;
using Core.Tokenization;
using Microsoft.Extensions.DependencyInjection;

namespace CLI;

public static class Registrations
{
    public static IServiceCollection AddDependencies(this IServiceCollection services)
    {
        services.AddSingleton<ITokenizer, BpeTokenizer>();
        services.AddSingleton<IDecoderFactory, TransformerDecoderFactory>();
        return services;
    }
}