package com.github.tjake.jlama.net;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.net.grpc.JlamaService;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

public class Coordinator {
    private static final Logger logger = LoggerFactory.getLogger(Coordinator.class);
    private final int port;
    private final int workerCount;
    private final Server server;
    private final AbstractModel model;
    private final JlamaService service;

    public Coordinator(Path modelPath, int port, int workerCount) {
        this.model = loadModel(AbstractModel.InferenceType.OUTPUT_TO_TOKEN, modelPath.toFile(), DType.F32, DType.I8, Optional.empty(), Optional.empty(), Optional.empty());
        this.port = port;
        this.workerCount = workerCount;
        this.service = new JlamaService(model, workerCount);
        this.server = ServerBuilder.forPort(port)
                .addService(service)
                .build();
    }

    public void start() throws IOException {
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime()
                .addShutdownHook(new Thread(() -> {
                    try {
                        this.stop();
                    } catch (InterruptedException e) {
                        logger.warn("Exception when shutting down", e);
                    }
                    logger.info("Server shut down");
                }));

        logger.info("Waiting for workers to register");
        service.waitForReady();
    }

    public void stop() throws InterruptedException {
        if (server != null) {
            service.shutdown();
            server.shutdownNow();
        }
    }

    public void generate(UUID session, String prompt, String cleanPrompt, float temperature, int ntokens, boolean useEOS, BiConsumer<String, Float> onTokenWithTimings) {
        service.waitForReady();

        long[] encoded = model.getTokenizer().encode(prompt);
        Preconditions.checkArgument(encoded.length < model.getConfig().contextLength);

        AbstractTensor logits = model.makeTensor(model.getConfig().vocabularySize);

        int[] promptTokens = new int[useEOS ? (1 + encoded.length + 1) : (1 + encoded.length)];

        promptTokens[0] = model.getConfig().bosToken;
        for (int i = 1; i < encoded.length; i++)
            promptTokens[i] = Ints.checkedCast(encoded[i]);

        int promptLength = encoded.length;

        if (useEOS) {
            promptTokens[promptTokens.length - 1] = model.getConfig().eosToken; //Add EOS
            promptLength++;
        }

        String clientPrompt = cleanPrompt == null ? prompt : cleanPrompt;
        onTokenWithTimings.accept(clientPrompt, 0f);
        long start = System.currentTimeMillis();

        AbstractTensor output = null;
        for (int i = 0; i < promptLength; i++) {
            output = service.generateNextOutput(session, promptTokens[i], i);
        }

        for (int i = promptLength; i < ntokens; i++) {
            int next = model.sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);

            //Model may tell us it's done
            if (next == model.getConfig().eosToken)
                break;

            try {
                String c = model.getTokenizer().decode(next);
                onTokenWithTimings.accept(c, (System.currentTimeMillis() - start) / (float) (i + 1));
            } catch (Exception e) {
                logger.error("Failed to decode token {}", next, e);
            }

            output.close();
            output = service.generateNextOutput(session, next, i);
        }
    }
}
