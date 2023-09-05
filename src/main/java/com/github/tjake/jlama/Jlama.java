package com.github.tjake.jlama;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.llama.LlamaConfig;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.SafeTensorIndex;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public class Jlama {

    static {
        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "24");
    }

    private static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, true);

    public static void main(String[] args) throws Exception {

        String prompt = args.length > 0 ? args[0] : "Simply put, the theory of relativity states that";

        String modelPrefix = "data/Llama-2-7b-chat-hf";
        try (SafeTensorIndex weights = SafeTensorIndex.loadWithWeights(Path.of(modelPrefix))) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer);

            model.generate(prompt, 0.6f, 10, false, makeOutHandler());
        }

    }

    public static BiConsumer<String, Float> makeOutHandler() {
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