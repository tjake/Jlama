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

import com.github.tjake.jlama.net.Worker;

import java.io.File;
import java.nio.file.Path;
import java.util.Optional;

import com.github.tjake.jlama.safetensors.DType;
import picocli.CommandLine;

@CommandLine.Command(name = "cluster-worker", description = "Connects to a cluster coordinator to perform distributed inference")
public class ClusterWorkerCommand extends BaseCommand {

    private static final Boolean useHostnameAsWorkerId = Boolean.getBoolean("jlama.use_hostname_as_workerid");
    private static final String HOSTNAME = System.getenv("HOSTNAME");

    @CommandLine.Option(names = { "-o", "--host" }, description = "hostname of coordinator", required = true)
    String host;

    @CommandLine.Option(names = { "-g",
        "--grpc-port" }, description = "grpc port to listen on (default: ${DEFAULT-VALUE})", defaultValue = "9777")
    int grpcPort = 9777;

    @CommandLine.Option(names = { "-w",
        "--worker-id" }, description = "consistent name to use when register this worker with the coordinator")
    String workerId = useHostnameAsWorkerId ? HOSTNAME : null;

    @CommandLine.Option(names = { "-mt", "--model-type"}, description = "The models base type F32/BF16 (default: ${DEFAULT-VALUE})", defaultValue = "F32")
    DType modelType = DType.F32;

    @Override
    public void run() {
        try {
            if (workerId != null) System.out.println("Using " + workerId + " as worker id");

            Path model = SimpleBaseCommand.getModel(modelName, modelDirectory, true, downloadSection.branch, downloadSection.authToken, false);

            Worker w = new Worker(
                model.toFile(),
                SimpleBaseCommand.getOwner(modelName),
                SimpleBaseCommand.getName(modelName),
                modelType,
                host,
                grpcPort,
                workingDirectory,
                workingMemoryType,
                workingQuantizationType,
                Optional.ofNullable(modelQuantization),
                Optional.ofNullable(workerId),
                    Optional.ofNullable(downloadSection.authToken),
                    Optional.ofNullable(downloadSection.branch)
            );

            w.run();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
