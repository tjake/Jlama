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

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.net.grpc.JlamaService;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.HTTPSafeTensorLoader;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiConsumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Coordinator implements Generator {
    private static final Logger logger = LoggerFactory.getLogger(Coordinator.class);
    private static final ConcurrentMap<UUID, Integer> sessionPositions = new ConcurrentHashMap<>();
    private final int port;
    private final int workerCount;
    private final Server server;
    private final AbstractModel model;
    private final JlamaService service;

    public Coordinator(
        File modelPath,
        String modelOwner,
        String modelName,
        DType modelDType,
        File workingDirectory,
        int port,
        int workerCount,
        Optional<String> authToken,
        Optional<String> branch
    ) {
        Preconditions.checkArgument(workerCount != 0 && ((workerCount & (workerCount - 1)) == 0), "worker count must be a power of 2");

        Function<File, WeightLoader> weightLoaderFunction = SafeTensorSupport.isModelLocal(modelPath.toPath())
            ? b -> SafeTensorSupport.loadWeights(modelPath)
            : b -> new HTTPSafeTensorLoader(modelPath.toPath(), modelOwner, modelName, modelDType, authToken, branch);

        this.model = loadModel(
            AbstractModel.InferenceType.OUTPUT_TO_TOKEN,
            modelPath,
            workingDirectory,
            DType.F32,
            DType.I8,
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            weightLoaderFunction
        );
        this.port = port;
        this.workerCount = workerCount;
        this.service = new JlamaService(model, workerCount);
        this.server = ServerBuilder.forPort(port).addService(service).build();
    }

    public Tokenizer getTokenizer() {
        return model.getTokenizer();
    }

    public Optional<PromptSupport> promptSupport() {
        return model.promptSupport();
    }

    public void start() throws IOException {
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                this.stop();
            } catch (InterruptedException e) {
                logger.warn("Exception when shutting down", e);
            }
            logger.info("Server shut down");
        }));

        logger.info("Waiting for {} workers to register", workerCount);
        service.waitForReady();
        logger.info("Coordinator ready with {} workers", workerCount);
    }

    public void stop() throws InterruptedException {
        if (server != null) {
            service.shutdown();
            server.shutdownNow();
        }
    }

    public float[] embed(String input, Generator.PoolingType poolingType) {
        throw new UnsupportedOperationException();
    }

    public Generator.Response generate(
        UUID session,
        PromptContext promptContext,
        float temperature,
        int ntokens,
        BiConsumer<String, Float> onTokenWithTimings
    ) {
        try {
            service.waitForReady();
            StringBuilder responseBuilder = new StringBuilder();
            StringBuilder responseWithSpecialTokens = new StringBuilder();

            int startPos = sessionPositions.computeIfAbsent(session, s -> 0);

            FinishReason finishReason = FinishReason.MAX_TOKENS;
            long[] encoded = model.getTokenizer().encode(promptContext.getPrompt());
            Preconditions.checkArgument(encoded.length < model.getConfig().contextLength);

            AbstractTensor logits = model.makeDenseTensor(model.getConfig().vocabularySize);

            Integer[] promptTokens = new Integer[1 + encoded.length];

            promptTokens[0] = model.getConfig().bosToken;
            for (int i = 1; i <= encoded.length; i++)
                promptTokens[i] = Ints.checkedCast(encoded[i - 1]);

            int promptLength = encoded.length;

            long start = System.currentTimeMillis();

            AbstractTensor output = service.generateNextOutput(session, Arrays.asList(promptTokens), startPos);

            long promptTime = System.currentTimeMillis();
            int lastPosition = startPos + promptLength;
            int tokensGenerated = 0;
            sessionPositions.put(session, lastPosition);
            for (int i = promptLength; i < ntokens; i++) {
                int next = model.sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);
                output.close();

                // Model may tell us it's done
                if (model.getConfig().eosTokens.contains(next)) {
                    finishReason = FinishReason.STOP_TOKEN;
                    break;
                }

                try {
                    String c = model.getTokenizer().decode(next);
                    if (model.getTokenizer().getModel().isSpecialToken(next)) {
                        responseWithSpecialTokens.append(c);
                    } else {
                        onTokenWithTimings.accept(c, (System.currentTimeMillis() - start) / (float) (i + 1));
                        responseBuilder.append(c);
                        responseWithSpecialTokens.append(c);
                    }
                } catch (Exception e) {
                    logger.error("Failed to decode token {}", next, e);
                }

                output = service.generateNextOutput(session, next, i);
                tokensGenerated++;
                sessionPositions.put(session, lastPosition++);
            }

            return new Generator.Response(
                responseBuilder.toString(),
                responseWithSpecialTokens.toString(),
                finishReason,
                promptTokens.length,
                tokensGenerated,
                promptTime - start,
                System.currentTimeMillis() - promptTime
            );
        } catch (Throwable t) {
            logger.warn("Error generating tokens for session {}", session, t);
            return new Generator.Response("", "", FinishReason.ERROR, 0, 0, 0, 0);
        }
    }
}
