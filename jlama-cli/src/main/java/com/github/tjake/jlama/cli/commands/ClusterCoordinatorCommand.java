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

import com.github.tjake.jlama.net.Coordinator;
import picocli.CommandLine;

@CommandLine.Command(name = "cluster-coordinator", description = "Starts a distributed rest api for a model using cluster workers")
public class ClusterCoordinatorCommand extends BaseCommand {

    @CommandLine.Option(names = { "-w", "--worker-count" }, description = "signifies this instance is a coordinator", required = true)
    int workerCount = 1;

    @CommandLine.Option(names = { "-g",
        "--grpc-port" }, description = "grpc port to listen on (default: ${DEFAULT-VALUE})", defaultValue = "9777")
    int grpcPort = 9777;

    @CommandLine.Option(names = { "-p",
        "--port" }, description = "http port to listen on (default: ${DEFAULT-VALUE})", defaultValue = "8080")
    int port = 8080;

    @Override
    public void run() {
        try {
            Coordinator c = new Coordinator(model, workingDirectory, grpcPort, workerCount);

            new Thread(() -> {
                try {
                    c.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            /*UndertowJaxrsServer ut = new UndertowJaxrsServer();
            ut.deploy(new JlamaRestApi(c), APPLICATION_PATH);
            ut.addResourcePrefixPath(
                    "/ui",
                    resource(new ClassPathResourceManager(ServeCommand.class.getClassLoader()))
                            .setDirectoryListingEnabled(true)
                            .addWelcomeFiles("index.html"));
            
            System.out.println("Chat UI: http://localhost:" + port + "/ui/index.html");
            ut.start(Undertow.builder().addHttpListener(port, "0.0.0.0"));*/

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
