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
package com.github.tjake.jlama.net;

import ch.qos.logback.classic.Level;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import org.junit.Assume;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

public class DistributedServiceTest {

    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
        ch.qos.logback.classic.Logger rootLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(
            ch.qos.logback.classic.Logger.ROOT_LOGGER_NAME
        );
        rootLogger.setLevel(Level.toLevel("info"));
    }

    @Test
    void manyWorkerTestLLama() throws Exception {
        Path modelRoot = Paths.get("../models/Meta-Llama-3-8B-Instruct-jlama-Q4");
        String modelName = "Meta-Llama-3-8B-Instruct-jlama-Q4";
        String modelOwner = "tjake";

        Assume.assumeTrue(Files.exists(modelRoot));

        Coordinator coordinator = new Coordinator(
            modelRoot.toFile(),
            modelOwner,
            modelName,
            DType.Q4,
            null,
            8888,
            4,
            true,
            true,
            Optional.empty(),
            Optional.empty()
        );
        try {
            new Thread(() -> {
                try {
                    coordinator.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            startWorker(modelRoot, modelOwner, modelName, 1);
            startWorker(modelRoot, modelOwner, modelName, 2);
            startWorker(modelRoot, modelOwner, modelName, 3);
            startWorker(modelRoot, modelOwner, modelName, 4);

            coordinator.generate(
                UUID.randomUUID(),
                PromptContext.of("Simply put, the theory of relativity states that"),
                0.3f,
                256,
                makeOutHandler()
            );
        } finally {
            coordinator.stop();
        }
    }

    private void startWorker(Path modelRoot, String modelOwner, String modelName, int workerNumber) throws Exception {

        new Thread(() -> {
            try {
                Worker worker = new Worker(
                    modelRoot.toFile(),
                    modelOwner,
                    modelName,
                    DType.Q4,
                    "localhost",
                    8888,
                    8888 + workerNumber,
                    null,
                    DType.F32,
                    DType.I8,
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty()
                );

                worker.run();
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                // worker.close();
            }
        }).start();
    }

    private BiConsumer<String, Float> makeOutHandler() {
        PrintWriter out;
        BiConsumer<String, Float> outCallback;
        if (System.console() == null) {
            AtomicInteger i = new AtomicInteger(0);
            StringBuilder b = new StringBuilder();
            out = new PrintWriter(System.out);
            outCallback = (w, t) -> {
                b.append(w);
                out.println(String.format("%d: %s [took %.2fms])", i.getAndIncrement(), b, t));
                out.flush();
            };
        } else {
            out = System.console().writer();
            outCallback = (w, t) -> {
                out.print(w);
                out.flush();
            };
        }

        return outCallback;
    }
}
