using CLI;
using CliFx;
using Microsoft.Extensions.DependencyInjection;

await new CliApplicationBuilder()
    .AddCommandsFromThisAssembly()
    .UseTypeActivator(commandTypes =>
    {
        var services = new ServiceCollection();

        services.AddDependencies();

        foreach (var commandType in commandTypes)
            services.AddTransient(commandType);

        return services.BuildServiceProvider();
    })
    .Build()
    .RunAsync();