package com.github.tjake.jlama.net;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.llama.LlamaConfig;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.WeightLoader;
import org.junit.Assume;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public class DistributedServiceTest {

    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    private static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, true);

    @Test
    void oneWorkerTestLLama() throws Exception {
        Path modelPath = Paths.get("../models/Llama-2-7b-chat-hf-jlama-Q4");
        Assume.assumeTrue(Files.exists(modelPath));

        Coordinator coordinator = new Coordinator(modelPath, 8888, 1);
        new Thread(() -> {
            try {
                coordinator.start();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();

        startWorker(modelPath);

        coordinator.generate(UUID.randomUUID(), "Simply put, the theory of relativity states that", null, 0.7f, 256, false, makeOutHandler());
    }

    @Test
    void manyWorkerTestLLama() throws Exception {
        Path modelRoot = Paths.get("../models/Llama-2-7b-chat-hf-jlama-Q4");
        Assume.assumeTrue(Files.exists(modelRoot));

        Coordinator coordinator = new Coordinator(modelRoot, 8888, 2);
        new Thread(() -> {
            try {
                coordinator.start();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();

        //startWorker(modelRoot);
        //startWorker(modelRoot);
        startWorker(modelRoot);
        startWorker(modelRoot);

        coordinator.generate(UUID.randomUUID(), "Simply put, the theory of relativity states that", null, 0.7f, 256, false, makeOutHandler());
    }




    private void startWorker(Path modelRoot) throws Exception {
        Worker worker = new Worker(modelRoot, "localhost", 8888, DType.F32, DType.I8, Optional.empty());
        new Thread(() -> {
            try {
                worker.run();
            } catch (Exception e) {
                e.printStackTrace();
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
            outCallback = (w,t) ->  {
                b.append(w);
                out.println(String.format("%d: %s [took %.2fms])", i.getAndIncrement(), b, t));
                out.flush();
            };
        } else {
            out = System.console().writer();
            outCallback = (w,t) -> {
                out.print(w);
                out.flush();
            };
        }

        return outCallback;
    }
}
