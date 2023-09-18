package com.github.tjake.jlama.models;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.bert.BertConfig;
import com.github.tjake.jlama.model.bert.BertModel;
import com.github.tjake.jlama.model.bert.BertTokenizer;
import com.github.tjake.jlama.safetensors.*;
import com.github.tjake.jlama.model.gpt2.GPT2Config;
import com.github.tjake.jlama.model.gpt2.GPT2Model;
import com.github.tjake.jlama.model.gpt2.GPT2Tokenizer;
import com.github.tjake.jlama.model.llama.LlamaConfig;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;

import org.junit.Assume;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public class TestModels {

    static {
        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "" + Math.max(4, Runtime.getRuntime().availableProcessors() / 2));
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    private static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, true);
    private static final Logger logger = LoggerFactory.getLogger(TestModels.class);

    @Test
    public void GPT2Run() throws IOException {
        String modelPrefix = "models/gpt2-medium";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
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
        String modelPrefix = "models/Llama-2-7b-chat-hf";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (SafeTensorIndex weights = SafeTensorIndex.loadWithWeights(Path.of(modelPrefix))) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer);

            String prompt = "Simply put, the theory of relativity states that";
            model.generate(prompt, 0.2f, 128, false, makeOutHandler());
        }
    }

    @Test
    public void TinyLlamaRun() throws Exception {
        String modelPrefix = "models/TinyLLama";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        try (RandomAccessFile sc = new RandomAccessFile(modelPrefix+"/model.safetensors", "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights weights = SafeTensors.readBytes(bb);
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer);

            String prompt = "Lily picked up a flower and gave it to";
            model.generate(prompt, 0.9f, 128, false, makeOutHandler());
        }
    }

    @Test
    public void BertRun() throws Exception {
        String modelPrefix = "models/e5-small-v2";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        try (RandomAccessFile sc = new RandomAccessFile(modelPrefix+"/model.safetensors", "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights weights = SafeTensors.readBytes(bb);
            Tokenizer tokenizer = new BertTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), BertConfig.class);
            BertModel model = new BertModel(c, weights, tokenizer);

            String base = "A man is eating food.";
            String[] examples = new String[]{
                    "A man is eating a piece of bread.",
                    "The girl is carrying a baby.",
                    "A man is riding a horse.",
                    "A woman is playing violin.",
                    "Two men pushed carts through the woods.",
                    "A man is riding a white horse on an enclosed ground.",
                    "A monkey is playing drums.",
                    "Someone in a gorilla costume is playing a set of drums."
            };

            float[] be = model.embed(base);
            logger.info("base is {}", base);
            float maxc = 0.0f;
            String bestMatch = "";
            for (int i = 0; i < examples.length; i++) {
                float vs = VectorMath.cosineSimilarity(be, model.embed(examples[i]));
                logger.info("vs {} => {}", examples[i], vs);
                if (vs > maxc) {
                    maxc = vs;
                    bestMatch = examples[i];
                }
            }

            logger.info("Best match for: '{}' is '{}'", base, bestMatch);

            long start = System.currentTimeMillis();
            VectorMath.pfor(0, 1000, i -> model.embed(base));
            long elapsed = System.currentTimeMillis() - start;
            logger.info("took {} seconds, {}ms per emb", elapsed/1000f, elapsed/1000f);

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
