package com.github.tjake.llmj.models;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.llmj.model.gpt2.GPT2Config;
import com.github.tjake.llmj.model.gpt2.GPT2Model;
import com.github.tjake.llmj.model.gpt2.GPT2Tokenizer;
import com.github.tjake.llmj.model.llama.LlamaConfig;
import com.github.tjake.llmj.model.llama.LlamaModel;
import com.github.tjake.llmj.model.llama.LlamaTokenizer;
import com.github.tjake.llmj.safetensors.*;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public class TestModels {

    private static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, true);
    private static final Logger logger = LoggerFactory.getLogger(TestModels.class);

    @Test
    public void GPT2Run() throws IOException {
        String modelPrefix = "data/gpt2-medium";
        try (RandomAccessFile sc = new RandomAccessFile(modelPrefix+"/model.safetensors", "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights v = SafeTensors.readBytes(bb);
            Tokenizer tokenizer = new GPT2Tokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), GPT2Config.class);
            GPT2Model gpt2 = new GPT2Model(c, v, tokenizer);

            String prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, " +
                    "previously unexplored valley, in the Andes Mountains. " +
                    "Even more surprising to the researchers was the fact that the unicorns spoke perfect English.";
            gpt2.generate(prompt, 0.6f, 256, false, makeOutHandler());
        }
    }

    @Test
    public void LlamaRun() throws Exception {
        String modelPrefix = "data/Llama-2-7b-chat-hf";
        try (SafeTensorIndex weights = SafeTensorIndex.loadWithWeights(Path.of(modelPrefix))) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer);

            String prompt = "Simply put, the theory of relativity states that";
            model.generate(prompt, 0.6f, 256, false, makeOutHandler());
        }
    }

    @Test
    public void TinyLlamaRun() throws Exception {
        String modelPrefix = "data/TinyLLama";

        try (RandomAccessFile sc = new RandomAccessFile(modelPrefix+"/model.safetensors", "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights weights = SafeTensors.readBytes(bb);
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer);

            String prompt = "Lily picked up a flower and gave it to";
            model.generate(prompt, 0.9f, 256, false, makeOutHandler());
        }
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
