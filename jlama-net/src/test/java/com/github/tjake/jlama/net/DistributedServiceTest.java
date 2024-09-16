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
    void oneWorkerTestLLama() throws Exception {
        Path modelPath = Paths.get("../models/Llama-2-7b-chat-hf-jlama-Q4");
        Assume.assumeTrue(Files.exists(modelPath));

        Coordinator coordinator = new Coordinator(modelPath.toFile(), com.google.common.io.Files.createTempDir(), 8888, 1);
        try {
            new Thread(() -> {
                try {
                    coordinator.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            startWorker(modelPath);

            coordinator.generate(
                UUID.randomUUID(),
                PromptContext.of("Simply put, the theory of relativity states that"),
                0.7f,
                256,
                makeOutHandler()
            );

        } finally {
            coordinator.stop();
        }
    }

    @Test
    void manyWorkerTestLLama() throws Exception {
        // Path modelRoot = Paths.get("../models/Mixtral-8x7B-Instruct-v0.1-jlama-Q4");
        Path modelRoot = Paths.get("../models/Meta-Llama-3.1-8B-Instruct-jlama-Q4");
        Assume.assumeTrue(Files.exists(modelRoot));

        Coordinator coordinator = new Coordinator(modelRoot.toFile(), null, 8888, 4);
        try {
            new Thread(() -> {
                try {
                    coordinator.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();

            startWorker(modelRoot);
            startWorker(modelRoot);
            startWorker(modelRoot);
            startWorker(modelRoot);

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

    private void startWorker(Path modelRoot) throws Exception {
        Worker worker = new Worker(modelRoot.toFile(), "localhost", 8888, null, DType.F32, DType.I8, Optional.empty(), Optional.empty());
        new Thread(() -> {
            try {
                worker.run();
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                worker.close();
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
