package com.github.tjake.jlama.cli.commands;

import java.nio.file.Path;

import com.github.tjake.jlama.cli.serve.JlamaRestApi;
import com.github.tjake.jlama.net.Coordinator;
import io.undertow.Undertow;
import io.undertow.server.handlers.resource.ClassPathResourceManager;
import org.jboss.resteasy.plugins.server.undertow.UndertowJaxrsServer;
import picocli.CommandLine;

import static com.github.tjake.jlama.cli.commands.ServeCommand.APPLICATION_PATH;
import static io.undertow.Handlers.resource;

@CommandLine.Command(name = "cluster-coordinator", description = "Starts a distributed rest api for a model using cluster workers")
public class ClusterCoordinatorCommand extends BaseCommand {
    
    @CommandLine.Option(names = {"-w", "--worker-count"}, description = "signifies this instance is a coordinator", defaultValue = "1")
    int workerCount = 1;

    @CommandLine.Option(names = {"-g", "--grpc-port"}, description = "grpc port to listen on", defaultValue = "9777")
    int grpcPort = 9777;

    @CommandLine.Option(names = {"-p", "--port"}, description = "http port to listen on", defaultValue = "8080")
    int port = 8080;

    @Override
    public void run()
    {
        try {
            Coordinator c = new Coordinator(model, workingDirectory, grpcPort, workerCount);

            new Thread(() -> {
                try {
                    c.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            UndertowJaxrsServer ut = new UndertowJaxrsServer();
            ut.deploy(new JlamaRestApi(c), APPLICATION_PATH);
            ut.addResourcePrefixPath("/ui", resource(new ClassPathResourceManager(ServeCommand.class.getClassLoader())).setDirectoryListingEnabled(true).addWelcomeFiles("index.html"));

            System.out.println("Chat UI: http://localhost:" + port + "/ui/index.html");
            ut.start(Undertow.builder().addHttpListener(port, "0.0.0.0"));

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
