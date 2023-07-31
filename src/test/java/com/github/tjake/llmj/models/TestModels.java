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

            PrintWriter out = System.console() == null ? new PrintWriter(System.out) : System.console().writer();
            String prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, " +
                    "previously unexplored valley, in the Andes Mountains. " +
                    "Even more surprising to the researchers was the fact that the unicorns spoke perfect English.";
            gpt2.run(prompt, 0.6f, 256, s -> {out.print(s); out.flush();});
        }
    }

    @Test
    public void LlamaRun() throws Exception {
        String modelPrefix = "data/llama2-7b-chat-hf";
        try (SafeTensorIndex weights = SafeTensorIndex.loadWithWeights(Path.of(modelPrefix))) {

            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer);

            PrintWriter out = System.console() == null ? new PrintWriter(System.out) : System.console().writer();
            String prompt = "Why did the chicken cross the road?";
            model.run(prompt, 0.6f, 256, false, s -> {out.print(s); out.flush();});
        }
    }
}
