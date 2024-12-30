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
package com.github.tjake.jlama.model;

import static com.github.tjake.jlama.util.JsonSupport.om;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.model.gemma.GemmaConfig;
import com.github.tjake.jlama.model.gemma.GemmaModel;
import com.github.tjake.jlama.model.gemma.GemmaTokenizer;
import com.github.tjake.jlama.model.llama.LlamaConfig;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.model.mistral.MistralConfig;
import com.github.tjake.jlama.model.mistral.MistralModel;
import com.github.tjake.jlama.model.mixtral.MixtralConfig;
import com.github.tjake.jlama.model.mixtral.MixtralModel;
import com.github.tjake.jlama.safetensors.*;
import com.github.tjake.jlama.safetensors.prompt.*;
import com.github.tjake.jlama.safetensors.tokenizer.BPETokenizer;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestModels {

    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "2");
        // System.setProperty("jlama.force_panama_tensor_operations", "true");
    }

    private static final Logger logger = LoggerFactory.getLogger(TestModels.class);

    @Test
    public void GPT2Run() throws IOException {
        String modelPrefix = "../models/gpt2-medium";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        AbstractModel gpt2 = ModelSupport.loadModel(new File(modelPrefix), DType.F32, DType.F32);
        PromptContext prompt = PromptContext.of(
            "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
                + "previously unexplored valley, in the Andes Mountains. "
                + "Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
        );

        gpt2.generate(UUID.randomUUID(), prompt, 0.8f, 256, makeOutHandler());
    }

    @Test
    public void Qwen2Run() throws IOException {
        String modelPrefix = "../models/Qwen_Qwen2.5-0.5B-Instruct-JQ4";    
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        AbstractModel qwen2 = ModelSupport.loadModel(new File(modelPrefix), DType.F32, DType.I8);

        int ntools = 200;
        Tool[] tools = new Tool[ntools];
        for (int i = 0; i < ntools; i++) {
            Tool tool = Tool.from(Function.builder()
                    .description("some tool "+i)
                    .addParameter("input", "string", "an input", true)
                    .name("some-tool-"+i)
                    .build());
            tools[i] = tool;
        }

        PromptContext prompt = qwen2.promptSupport().get().builder()
                .addSystemMessage("You are a helpful chatbot who writes short responses.")
                .addUserMessage("What is the capital of France?")
                .build(tools);

        Generator.Response r = qwen2.generate(UUID.randomUUID(), prompt, 0.9f, 32 * 1024, makeOutHandler());
        logger.info("Response: {}", r);
    }

    @Test
    public void Gemma2Run() throws IOException {
        String modelPrefix = "../models/google_gemma-2-2b-it";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        AbstractModel gemma2 = ModelSupport.loadModel(new File(modelPrefix), DType.F32, DType.F32);
        PromptContext prompt = gemma2.promptSupport().get().builder().addUserMessage("What is the capital of France?").build();

        Generator.Response r = gemma2.generate(UUID.randomUUID(), prompt, 0.0f, 1024, makeOutHandler());
        logger.info("Response: {}", r);
    }

    @Test
    public void GraniteRun() throws IOException {
        String modelPrefix = "../models/ibm-granite_granite-3.0-2b-instruct";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        AbstractModel granite = ModelSupport.loadModel(new File(modelPrefix), DType.F32, DType.F32);
        PromptContext prompt = granite.promptSupport().get().builder().addUserMessage("What is the capital of France?").build();

        Generator.Response r = granite.generate(UUID.randomUUID(), prompt, 0.0f, 1024, makeOutHandler());
        logger.info("Response: {}", r);
    }

    @Test
    public void LlamaRun() throws Exception {
        String modelPrefix = "../models/tjake_Llama-3.2-1B-Instruct-Jlama-Q4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights = SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.F32, Optional.empty());

            PromptSupport.Builder builder = model.promptSupport().get().builder();

            builder.addUserMessage("What is the temp in paris right now?");
            builder.addGenerationPrompt(true);

            Tool t = Tool.from(
                Function.builder()
                    .name("get_current_temperature")
                    .description("Simulates getting the current temperature at a location.")
                    .addParameter("location", "string", "The location to get the temperature for, in the format \"City, Country\".", true)
                    .addParameter("unit", "string", "The unit to return the temperature in (e.g., \"celsius\", \"fahrenheit\").", true)
                    .build()
            );

            PromptContext promptContext = builder.build(t);

            logger.info("First prompt \n{}", promptContext);
            Generator.Response r = model.generate(UUID.randomUUID(), promptContext, 0.0f, 1024, (l, f) -> {});
            logger.info("Response: {}", r.responseText);

            Assert.assertEquals(Generator.FinishReason.TOOL_CALL, r.finishReason);
            Assert.assertEquals(1, r.toolCalls.size());
            Assert.assertEquals("get_current_temperature", r.toolCalls.get(0).getName());

            ToolCall f = r.toolCalls.get(0);
            logger.info("Calling tool: {}", f.getName());

            builder.addToolCall(f);
            builder.addToolResult(ToolResult.from(f.getName(), null, 20f));
            logger.info("Second prompt {}", builder.build());
            Generator.Response r2 = model.generate(UUID.randomUUID(), builder.build(), 0.0f, 1024, (l, p) -> {});

            Assert.assertTrue(r2.responseText, r2.responseText.contains("20"));
            logger.info("Response: {}", r2.responseText);
        }
    }

    @Test
    public void DeepCoderRun() throws Exception {
        String modelPrefix = "../models/deepseek-coder-1.3b-base";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights = SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.F32, Optional.empty());
            String prompt = "#write a quicksort algorithm in python";
            model.generate(UUID.randomUUID(), PromptContext.of(prompt), 0.7f, 256, makeOutHandler());
        }
    }

    @Test
    public void MistralRun() throws Exception {
        String modelPrefix = "../models/tjake_Mistral-7B-Instruct-v0.3-JQ4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights = SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            BPETokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), MistralConfig.class);
            MistralModel model = new MistralModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());

            PromptSupport.Builder builder = model.promptSupport().get().builder();

            builder.addUserMessage("What is the temp in paris right now?");
            builder.addGenerationPrompt(true);

            Tool t = Tool.from(
                Function.builder()
                    .name("get_current_temperature")
                    .description("Simulates getting the current temperature at a location.")
                    .addParameter("location", "string", "The location to get the temperature for, in the format \"City, Country\".", true)
                    .addParameter("unit", "string", "The unit to return the temperature in (e.g., \"celsius\", \"fahrenheit\").", true)
                    .build()
            );

            PromptContext promptContext = builder.build(t);

            logger.info("First prompt \n{}", promptContext);

            Generator.Response r = model.generate(UUID.randomUUID(), promptContext, 0.0f, 1024, makeOutHandler());

            logger.info("Response: {}", r);
            Assert.assertEquals(Generator.FinishReason.TOOL_CALL, r.finishReason);
            Assert.assertEquals(1, r.toolCalls.size());
            Assert.assertEquals("get_current_temperature", r.toolCalls.get(0).getName());

            ToolCall f = r.toolCalls.get(0);

            logger.info("Calling tool: {}", f.getName());

            builder.addToolCall(f);
            builder.addToolResult(ToolResult.from(f.getName(), f.getId(), 20f));

            logger.info("Second prompt {}", builder.build());
            Generator.Response r2 = model.generate(UUID.randomUUID(), builder.build(), 0.0f, 1024, makeOutHandler());

            Assert.assertTrue(r2.responseText, r2.responseText.contains("20"));

            logger.info("Response: {}", r2);
        }
    }

    @Test
    public void MixtralRun() throws Exception {
        String modelPrefix = "../models/Mixtral-8x7B-Instruct-v0.1-jlama-Q4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights = SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            MixtralConfig c = om.readValue(new File(modelPrefix + "/config.json"), MixtralConfig.class);
            MixtralModel model = new MixtralModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());
            String prompt0 =
                "Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, "
                    + "allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, "
                    + "or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance. Explain the above in one sentence:";

            String prompt = "Tell me a joke.";
            PromptContext p = model.promptSupport().get().builder().addUserMessage(prompt).build();

            model.generate(UUID.randomUUID(), p, 0.7f, 256, makeOutHandler());
        }
    }

    @Test
    public void GemmaRun() throws Exception {
        String modelPrefix = "../models/google_codegemma-2b-JQ4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights = SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            Tokenizer tokenizer = new GemmaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), GemmaConfig.class);
            AbstractModel model = new GemmaModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());
            PromptContext p = PromptContext.of("<|fim_prefix|>public void fibonocci(int n) {<|fim_suffix|><|fim_middle|>");
            model.generate(UUID.randomUUID(), p, 0.3f, c.contextLength, makeOutHandler());
        }
    }

    @Test
    public void YiCoderRun() throws Exception {
        String modelPrefix = "../models/01-ai_Yi-Coder-9B-Chat-JQ4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights = SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            LlamaConfig c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());
            String prompt = "Write a java function that takes a list of integers and returns the sum of all the integers in the list.";
            PromptContext p = model.promptSupport().get().builder().addUserMessage(prompt).build();
            model.generate(UUID.randomUUID(), p, 0.7f, c.contextLength, makeOutHandler());
        }
    }

    @Test
    public void testQuantize() throws Exception {
        String modelPrefix = "models/Llama-2-7b-chat-hf";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Path tmpOut = Files.createTempDirectory("jltest");
        try {
            Path out = SafeTensorSupport.quantizeModel(
                Paths.get(modelPrefix),
                DType.Q4,
                new String[] { "model.embed_tokens.weight", "lm_head.weight", },
                null,
                Optional.of(tmpOut)
            );

            Assert.assertEquals(tmpOut, out);

            WeightLoader weights = SafeTensorSupport.loadWeights(tmpOut.toFile());
            LlamaTokenizer tokenizer = new LlamaTokenizer(tmpOut);
            Config c = om.readValue(new File(tmpOut + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());

            String prompt = "Lily picked up a flower and gave it to";
            model.generate(UUID.randomUUID(), PromptContext.of(prompt), 0.7f, 128, makeOutHandler());
        } finally {
            Arrays.stream(Objects.requireNonNull(tmpOut.toFile().listFiles())).forEach(f -> {
                try {
                    Files.delete(f.toPath());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });

            Files.deleteIfExists(tmpOut);
        }
    }

    @Test
    public void TinyLlamaRun() throws Exception {
        String modelPrefix = "../models/TinyLlama-1.1B-Chat-v1.0-jlama-Q4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        AbstractModel model = ModelSupport.loadModel(new File(modelPrefix), DType.F32, DType.I8);

        String prompt = "What is the best season to plant avocados?";
        PromptContext promptContext = model.promptSupport()
            .get()
            .builder()
            .addSystemMessage("You are a helpful chatbot who writes short responses.")
            .addUserMessage(prompt)
            .build();
        model.generate(UUID.randomUUID(), promptContext, 0.0f, 256, makeOutHandler());
    }

    @Test
    public void BertRun() throws Exception {
        String modelPrefix = "../models/e5-small-v2";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        AbstractModel model = ModelSupport.loadModel(new File(modelPrefix), DType.F32, DType.F32);

        String base = "A man is eating food.";
        String[] examples = new String[] { "A man is eating a piece of bread.", "The girl is carrying a baby.", "A man is riding a horse.",
            "A woman is playing violin.", "Two men pushed carts through the woods.", "A man is riding a white horse on an enclosed ground.",
            "A monkey is playing drums.", "Someone in a gorilla costume is playing a set of drums." };

        float[] be = model.embed(base, Generator.PoolingType.AVG);
        logger.info("base is {}", base);
        float maxc = 0.0f;
        String bestMatch = "";
        for (int i = 0; i < examples.length; i++) {
            float vs = VectorMath.cosineSimilarity(be, model.embed(examples[i], Generator.PoolingType.AVG));
            logger.info("vs {} => {}", examples[i], vs);
            if (vs > maxc) {
                maxc = vs;
                bestMatch = examples[i];
            }
        }

        logger.info("Best match for: '{}' is '{}'", base, bestMatch);

        long start = System.currentTimeMillis();
        VectorMath.pfor(0, 1000, i -> model.embed(base, Generator.PoolingType.AVG));
        long elapsed = System.currentTimeMillis() - start;
        logger.info("took {} seconds, {}ms per emb", elapsed / 1000f, elapsed / 1000f);
    }

    private BiConsumer<String, Float> makeOutHandler() {
        PrintWriter out;
        BiConsumer<String, Float> outCallback;

        AtomicInteger i = new AtomicInteger(0);
        StringBuilder b = new StringBuilder();
        out = new PrintWriter(System.out);
        outCallback = (w, t) -> {
            b.append(w);
            out.println(String.format("%d: %s [took %.2fms])", i.getAndIncrement(), b, t));
            out.flush();
        };

        return outCallback;
    }
}
