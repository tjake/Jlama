package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.cli.serve.JlamaRestApi;
import com.github.tjake.jlama.model.AbstractModel;
import io.undertow.Undertow;
import io.undertow.server.handlers.resource.ClassPathResourceManager;
import org.jboss.resteasy.plugins.server.undertow.UndertowJaxrsServer;
import picocli.CommandLine;

import java.util.Optional;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;
import static io.undertow.Handlers.resource;

@CommandLine.Command(name = "serve", description = "Starts a rest api for interacting with this model")
public class ServeCommand extends BaseCommand {

    @CommandLine.Option(names = {"-p", "--port"}, description = "http port (default: ${DEFAULT-VALUE})", defaultValue = "8080")
    int port = 8080;

    static final String APPLICATION_PATH = "/api";

    @Override
    public void run()
    {
        try {
            AbstractModel m = loadModel(model, workingDirectory, workingMemoryType, workingQuantizationType, java.util.Optional.ofNullable(modelQuantization), Optional.ofNullable(threadCount));

            UndertowJaxrsServer ut = new UndertowJaxrsServer();
            ut.deploy(new JlamaRestApi(m), APPLICATION_PATH);
            ut.addResourcePrefixPath("/ui", resource(new ClassPathResourceManager(ServeCommand.class.getClassLoader())).setDirectoryListingEnabled(true).addWelcomeFiles("index.html"));

            System.out.println("Chat UI: http://localhost:" + port + "/ui/index.html");
            ut.start(Undertow.builder().addHttpListener(port, "0.0.0.0"));
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
