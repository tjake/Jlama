package com.github.tjake.jlama.cli.commands;

import java.util.Optional;
import java.util.Properties;

import com.github.tjake.jlama.cli.serve.JlamaRestApi;
import com.github.tjake.jlama.net.Coordinator;
import com.github.tjake.jlama.net.Worker;
import io.undertow.Undertow;
import io.undertow.server.handlers.resource.ClassPathResourceManager;
import org.jboss.resteasy.plugins.server.undertow.UndertowJaxrsServer;
import picocli.CommandLine;

import static com.github.tjake.jlama.cli.commands.ServeCommand.APPLICATION_PATH;
import static io.undertow.Handlers.resource;

@CommandLine.Command(name = "cluster-worker", description = "Connects to a cluster coordinator to perform distributed inference")
public class ClusterWorkerCommand extends BaseCommand {

    private static final Boolean useHostnameAsWorkerId = Boolean.getBoolean("jlama.use_hostname_as_workerid");
    private static final String HOSTNAME = System.getenv("HOSTNAME");

    @CommandLine.Option(names = {"-o", "--host"}, description = "hostname of coordinator", required = true)
    String host;

    @CommandLine.Option(names = {"-g", "--grpc-port"}, description = "grpc port to listen on (default: ${DEFAULT-VALUE})", defaultValue = "9777")
    int grpcPort = 9777;

    @CommandLine.Option(names = {"-w", "--worker-id"}, description = "consistent name to use when register this worker with the coordinator")
    String workerId = useHostnameAsWorkerId ? HOSTNAME : null;

    @Override
    public void run()
    {
        try {
            if (workerId != null)
                System.out.println("Using " + workerId + " as worker id");
            Worker w = new Worker(model, host, grpcPort, workingDirectory, workingMemoryType, workingQuantizationType, Optional.ofNullable(modelQuantization), Optional.ofNullable(workerId));
            w.run();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
