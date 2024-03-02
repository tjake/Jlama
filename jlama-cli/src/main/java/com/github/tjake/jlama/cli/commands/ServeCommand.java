/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.cli.commands;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;
import static io.undertow.Handlers.resource;

import com.github.tjake.jlama.cli.serve.JlamaRestApi;
import com.github.tjake.jlama.model.AbstractModel;
import io.undertow.Undertow;
import io.undertow.server.handlers.resource.ClassPathResourceManager;
import java.util.Optional;
import org.jboss.resteasy.plugins.server.undertow.UndertowJaxrsServer;
import picocli.CommandLine;

@CommandLine.Command(name = "serve", description = "Starts a rest api for interacting with this model")
public class ServeCommand extends BaseCommand {

    @CommandLine.Option(
            names = {"-p", "--port"},
            description = "http port (default: ${DEFAULT-VALUE})",
            defaultValue = "8080")
    int port = 8080;

    static final String APPLICATION_PATH = "/api";

    @Override
    public void run() {
        try {
            AbstractModel m = loadModel(
                    model,
                    workingDirectory,
                    workingMemoryType,
                    workingQuantizationType,
                    java.util.Optional.ofNullable(modelQuantization),
                    Optional.ofNullable(threadCount));

            UndertowJaxrsServer ut = new UndertowJaxrsServer();
            ut.deploy(new JlamaRestApi(m), APPLICATION_PATH);
            ut.addResourcePrefixPath(
                    "/ui",
                    resource(new ClassPathResourceManager(ServeCommand.class.getClassLoader()))
                            .setDirectoryListingEnabled(true)
                            .addWelcomeFiles("index.html"));

            System.out.println("Chat UI: http://localhost:" + port + "/ui/index.html");
            ut.start(Undertow.builder().addHttpListener(port, "0.0.0.0"));
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
