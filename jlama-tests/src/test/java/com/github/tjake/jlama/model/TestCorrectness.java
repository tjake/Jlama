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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.gemma.GemmaTokenizer;
import com.github.tjake.jlama.model.gpt2.GPT2Tokenizer;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.safetensors.prompt.*;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.safetensors.tokenizer.WordPieceTokenizer;
import com.github.tjake.jlama.util.JsonSupport;
import com.google.common.io.Resources;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

public class TestCorrectness {

    @Test
    public void testBPETokenizer() {
        String modelPrefix = "../models/Llama-2-7b-chat-hf";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));

        String p = "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[] { 518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580,
            29962, 673, 29871 };

        Assert.assertArrayEquals(expected, actual);

        System.out.println(tokenizer.decode(actual));
    }

    @Test
    public void TestLLamaTokenizer() throws IOException {
        String modelPrefix = "../models/Llama-2-7b-chat-hf-2";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));

        String p = "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[] { 518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580,
            29962, 673, 29871 };

        Assert.assertArrayEquals(expected, actual);

        String out = tokenizer.decode(actual);
        Assert.assertEquals(p, out);

        String s = tokenizer.decode(518);
        Assert.assertEquals(" [", s);

        long[] token = tokenizer.encode(p + "\n");
        expected = new long[] { 518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580, 29962,
            673, 29871, 13 };
        Assert.assertArrayEquals(expected, token);
    }

    @Test
    public void TestRope() throws IOException {
        double[] expected = new double[] { 8.4147e-01, 7.6172e-01, 6.8156e-01, 6.0469e-01, 5.3317e-01, 4.6795e-01, 4.0931e-01, 3.5711e-01,
            3.1098e-01, 2.7043e-01, 2.3492e-01, 2.0391e-01, 1.7689e-01, 1.5338e-01, 1.3296e-01, 1.1522e-01, 9.9833e-02, 8.6488e-02,
            7.4919e-02, 6.4893e-02, 5.6204e-02, 4.8678e-02, 4.2157e-02, 3.6509e-02, 3.1618e-02, 2.7381e-02, 2.3712e-02, 2.0534e-02,
            1.7782e-02, 1.5399e-02, 1.3335e-02, 1.1548e-02, 9.9998e-03, 8.6595e-03, 7.4989e-03, 6.4938e-03, 5.6234e-03, 4.8697e-03,
            4.2170e-03, 3.6517e-03, 3.1623e-03, 2.7384e-03, 2.3714e-03, 2.0535e-03, 1.7783e-03, 1.5399e-03, 1.3335e-03, 1.1548e-03,
            1.0000e-03, 8.6596e-04, 7.4989e-04, 6.4938e-04, 5.6234e-04, 4.8697e-04, 4.2170e-04, 3.6517e-04, 3.1623e-04, 2.7384e-04,
            2.3714e-04, 2.0535e-04, 1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04 };

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(128, 4096 * 2, 10000.0, 1.0);

        for (int i = 0; i < 64; i++)
            Assert.assertEquals(expected[i], ropeFreqs[i + 64][1], 0.0001);

        expected = new double[] { 0.9200, -0.9031, -0.7639, -0.6592, -0.9904, -0.2474, 0.9597, -0.9819, 0.9835, -0.9696, 0.5065, 0.5448,
            -0.9266, -0.4176, 0.7772, 0.8945, 0.1165, -0.6750, -0.9962, -0.8492, -0.4416, 0.0250, 0.4284, 0.7205, 0.8991, 0.9835, 0.9986,
            0.9673, 0.9078, 0.8336, 0.7536, 0.6736, 0.5972, 0.5263, 0.4617, 0.4037, 0.3522, 0.3066, 0.2666, 0.2316, 0.2010, 0.1744, 0.1512,
            0.1310, 0.1136, 0.0984, 0.0852, 0.0738, 0.0640, 0.0554, 0.0480, 0.0415, 0.0360, 0.0312, 0.0270, 0.0234, 0.0202, 0.0175, 0.0152,
            0.0131, 0.0114, 0.0099, 0.0085, 0.0074 };

        for (int i = 0; i < 64; i++)
            Assert.assertEquals(expected[i], ropeFreqs[i + (64 * 64)][1], 0.0001);
    }

    @Test
    public void testRope2() throws IOException {
        List<List<Float>> real = om.readerFor(new TypeReference<ArrayList<ArrayList<Float>>>() {
        }).readValue(Resources.getResource("real.json"));
        List<List<Float>> imag = om.readerFor(new TypeReference<ArrayList<ArrayList<Float>>>() {
        }).readValue(Resources.getResource("imag.json"));

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(128, 2048, 10000.0, 1.0);

        Assert.assertEquals(imag.size(), real.size());
        Assert.assertEquals(ropeFreqs.length, real.size() * 64);

        for (int i = 0; i < real.size(); i++) {
            for (int j = 0; j < 64; j++) {
                Assert.assertEquals(real.get(i).get(j), ropeFreqs[i * 64 + j][0], 0.0001);
                Assert.assertEquals(imag.get(i).get(j), ropeFreqs[i * 64 + j][1], 0.0001);
            }
        }
    }

    @Test
    public void testFloatTypes() {
        for (int i = 0; i < 1000; i++) {
            float f = ThreadLocalRandom.current().nextFloat(-1, 1);
            Assert.assertEquals(f, FloatConversions.bFloat16ToFloat32(FloatConversions.float32ToBFloat16(f)), 0.01);
            Assert.assertEquals(f, Float.float16ToFloat(Float.floatToFloat16(f)), 0.001);
            Assert.assertEquals(f, FloatConversions.float16ToFloat32Alt(Float.floatToFloat16(f)), 0.001);
        }
    }

    @Test
    public void testGptTokenizer() throws IOException {
        String modelPrefix = "../models/gpt2-medium";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new GPT2Tokenizer(Paths.get(modelPrefix));

        String p = "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[] { 58, 38604, 60, 14026, 502, 257, 9707, 13, 12520, 238, 109, 46581, 38604, 60, 23998, 220 };

        String d = tokenizer.decode(actual);
        System.out.println(d);

        Assert.assertArrayEquals(expected, actual);
        Assert.assertEquals(p, d);
    }

    @Test
    public void testNemoTokenizer() throws IOException {
        String modelPrefix = "../models/Mistral-Nemo-Instruct-2407";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new GPT2Tokenizer(Paths.get(modelPrefix));

        String p = "Hello!";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[] { 22177, 1033 };

        String d = tokenizer.decode(actual);
        System.out.println(d);

        Assert.assertArrayEquals(expected, actual);
        Assert.assertEquals(p, d);
    }

    @Test
    public void testNeoTokenizer() throws IOException {
        String modelPrefix = "../models/deepseek-coder-1.3b-base";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));

        String p = "#write a quicksort algorithm";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[] { 2, 6449, 245, 3383, 3724, 6713 };

        String d = tokenizer.decode(actual);
        System.out.println(d);

        Assert.assertArrayEquals(expected, actual);
        Assert.assertEquals(p, d);
    }

    @Test
    public void testBertTokenizer() throws IOException {
        String modelPrefix = "../models/e5-small-v2";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new WordPieceTokenizer(Paths.get(modelPrefix));

        String p = "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[] { 101, 1031, 16021, 2102, 1033, 2425, 2033, 1037, 8257, 1012, 100, 1031, 1013, 16021, 2102, 1033, 3437,
            102 };

        String decodeExpected = "[CLS] [ inst ] tell me a joke. [UNK] [ / inst ] answer [SEP]";

        String decodeActual = tokenizer.decode(actual);
        System.out.println(decodeActual);

        Assert.assertArrayEquals(expected, actual);
    }

    @Test
    public void testGemmaTokenizer() throws IOException {
        String modelPrefix = "../models/gemma-7b";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new GemmaTokenizer(Paths.get(modelPrefix));

        String p = "Write me a poem about Machine Learning.";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[] { 5559, 682, 476, 19592, 1105, 13403, 14715, 235265 };

        String d = tokenizer.decode(actual);
        System.out.println(d);

        Assert.assertArrayEquals(expected, actual);
        Assert.assertEquals(p, d);
    }

    @Test
    public void testPromptSupport() {
        String modelPrefix = "../models/zephyr-7b-beta";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
        PromptSupport.Builder builder = tokenizer.promptSupport().get().builder();
        builder.addSystemMessage("You are a friendly chatbot who always responds in the style of a pirate");
        builder.addUserMessage("How many helicopters can a human eat in one sitting?");

        PromptContext prompt = builder.build();
        Assert.assertEquals(
            "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\nHow many helicopters can a human eat in one sitting?</s>\n<|assistant|>\n",
            prompt
        );

        long[] encoded = tokenizer.encode(prompt.getPrompt());
        long[] expected = new long[] { 523, 28766, 6574, 28766, 28767, 13, 1976, 460, 264, 10131, 10706, 10093, 693, 1743, 2603, 3673, 297,
            272, 3238, 302, 264, 17368, 380, 2, 28705, 13, 28789, 28766, 1838, 28766, 28767, 13, 5660, 1287, 19624, 410, 1532, 541, 264,
            2930, 5310, 297, 624, 6398, 28804, 2, 28705, 13, 28789, 28766, 489, 11143, 28766, 28767, 13 };

        String out = tokenizer.decode(encoded);
        Assert.assertEquals(prompt, out);
        Assert.assertArrayEquals(expected, encoded);
    }

    @Test
    public void testPromptSupport2() {
        String modelPrefix = "../models/Meta-Llama-3-8B-Instruct-jlama-Q4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
        PromptSupport.Builder builder = tokenizer.promptSupport().get().builder();
        builder.addSystemMessage("You are a friendly chatbot who always responds in the style of a pirate.");
        builder.addUserMessage("How many helicopters can a human eat in one sitting?");

        PromptContext prompt = builder.build();
        Assert.assertEquals(
            "<|start_header_id|>system<|end_header_id|>\n\n"
                + "You are a friendly chatbot who always responds in the style of a pirate.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                + "How many helicopters can a human eat in one sitting?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            prompt.getPrompt()
        );

        long[] encoded = tokenizer.encode(prompt.getPrompt());
        long[] expected = new long[] { 128000, 128006, 9125, 128007, 271, 2675, 527, 264, 11919, 6369, 6465, 889, 2744, 31680, 304, 279,
            1742, 315, 264, 55066, 128009, 128006, 882, 128007, 271, 4438, 1690, 59432, 649, 264, 3823, 8343, 304, 832, 11961, 30, 128009,
            128006, 78191, 128007, 271 };

        String out = tokenizer.decode(encoded);
        Assert.assertEquals(prompt, out);
        Assert.assertArrayEquals(expected, encoded);
    }

    @Test
    public void testPromptSupportWithTools() {
        String modelPrefix = "../models/Meta-Llama-3.1-8B-Instruct";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
        PromptSupport.Builder builder = tokenizer.promptSupport().get().builder();
        builder.addSystemMessage("You always respond as a pirate");
        builder.addUserMessage("What is the weather in paris right now?");
        builder.addGenerationPrompt(true);

        Tool t = Tool.from(
            Function.builder()
                .name("get_temperature")
                .description("Simulates getting the current temperature at a location.")
                .addParameter("location", "string", "The location to get the temperature for, in the format \"City, Country\".", true)
                .addParameter("unit", "string", "The unit to return the temperature in (e.g., \"celsius\", \"fahrenheit\").", true)
                .build()
        );

        // builder.addToolCall(new ToolCall("get_temperature", Map.of("location", "paris, france", "unit", "celsius")));

        // builder.addToolResult(ToolResult.from("get_temperature", "", Map.of("temperature", 25.0, "unit",
        // "celsius")));

        PromptContext prompt = builder.build(t);

        System.err.println(prompt);

        long[] encoded = tokenizer.encode(prompt.getPrompt());
        long[] expected = new long[] { 128006, 9125, 128007, 271, 13013, 25, 6125, 27993, 198, 38766, 1303, 33025, 2696, 25, 6790, 220,
            2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 2744, 6013, 439, 264, 55066, 128009, 128006, 882,
            128007, 271, 22818, 279, 2768, 5865, 11, 4587, 6013, 449, 264, 4823, 369, 264, 734, 1650, 449, 1202, 6300, 6105, 430, 1888,
            11503, 279, 2728, 10137, 382, 66454, 304, 279, 3645, 5324, 609, 794, 734, 836, 11, 330, 14105, 794, 11240, 315, 5811, 836, 323,
            1202, 907, 7966, 5519, 539, 1005, 7482, 382, 5018, 1337, 794, 330, 1723, 498, 330, 1723, 794, 5324, 609, 794, 330, 456, 54625,
            498, 330, 4789, 794, 330, 14354, 24031, 3794, 279, 1510, 9499, 520, 264, 3813, 10684, 330, 14105, 794, 5324, 1337, 794, 330,
            1735, 498, 330, 13495, 794, 5324, 2588, 794, 5324, 1337, 794, 330, 928, 498, 330, 4789, 794, 330, 791, 3813, 311, 636, 279,
            9499, 369, 11, 304, 279, 3645, 330, 13020, 11, 14438, 66820, 2186, 330, 3928, 794, 5324, 1337, 794, 330, 928, 498, 330, 4789,
            794, 330, 791, 5089, 311, 471, 279, 9499, 304, 320, 68, 1326, 2637, 330, 66, 41347, 498, 330, 69, 49010, 1865, 9388, 2186, 330,
            6413, 794, 4482, 2588, 498, 330, 3928, 1365, 3500, 633, 3923, 374, 279, 9282, 304, 41958, 1314, 1457, 30, 128009, 128006, 78191,
            128007, 271 };

        String out = tokenizer.decode(encoded);
        Assert.assertEquals(prompt, out);
        Assert.assertArrayEquals(expected, encoded);
    }

    @Test
    public void testMistralTools() {

        String modelPrefix = "../models/tjake_Mistral-7B-Instruct-v0.3-JQ4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
        PromptSupport.Builder builder = tokenizer.promptSupport().get().builder();
        builder.addSystemMessage("You always respond as a pirate");
        builder.addUserMessage("What is the weather in paris right now?");
        builder.addGenerationPrompt(true);

        Tool t = Tool.from(
            Function.builder()
                .name("get_current_temperature")
                .description("Simulates getting the current temperature at a location.")
                .addParameter("location", "string", "The location to get the temperature for, in the format \"City, Country\".", true)
                .addParameter("unit", "string", "The unit to return the temperature in (e.g., \"celsius\", \"fahrenheit\").", true)
                .build()
        );

        PromptContext prompt = builder.build(t);
        long[] encoded = tokenizer.encode(prompt.getPrompt());

        long[] official = new long[] { 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 1295, 29498,
            3790, 29498, 29475, 17329, 1316, 1113, 7286, 2032, 1113, 8322, 27653, 3487, 1040, 2636, 8409, 1206, 1032, 5491, 9959, 1113,
            12206, 2032, 10598, 1891, 2032, 1113, 3582, 1316, 1113, 11491, 2032, 10598, 3501, 2032, 10598, 1891, 2032, 1113, 2195, 1316,
            1113, 7286, 2032, 1113, 1782, 5491, 1066, 1393, 1040, 8409, 1122, 29493, 1065, 1040, 5800, 12547, 22781, 29493, 13776, 5651,
            1379, 1649, 1113, 6074, 2032, 10598, 1891, 2032, 1113, 2195, 1316, 1113, 7286, 2032, 1113, 1782, 5796, 1066, 1372, 1040, 8409,
            1065, 1093, 29474, 29491, 29489, 2831, 12547, 29485, 1958, 3938, 23102, 12547, 29490, 19425, 13075, 29524, 4913, 29507, 11549,
            1113, 11661, 2032, 8135, 3501, 1316, 1113, 6074, 3010, 1743, 10925, 7, 3, 1763, 2511, 10189, 1158, 1032, 18136, 1148, 781, 781,
            3963, 1117, 1040, 8854, 1065, 1708, 1046, 1871, 1823, 29572, 4 };

        for (int i = 0; i < official.length; i++) {
            if (official[i] != encoded[i]) {
                System.out.println(i + " " + official[i] + " " + encoded[i]);
            } else System.out.println(i + " " + official[i]);
        }

        Assert.assertArrayEquals(official, encoded);

        String expected =
            "[AVAILABLE_TOOLS] [{\"type\": \"function\", \"function\": {\"name\": \"get_current_temperature\", \"description\": \"Simulates getting the current temperature at a location.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The location to get the temperature for, in the format \\\"City, Country\\\".\"}, \"unit\": {\"type\": \"string\", \"description\": \"The unit to return the temperature in (e.g., \\\"celsius\\\", \\\"fahrenheit\\\").\"}}, \"required\": [\"location\", \"unit\"]}}}][/AVAILABLE_TOOLS][INST] You always respond as a pirate\n"
                + "\n"
                + "What is the weather in paris right now?[/INST]";

        Assert.assertEquals(expected, prompt);
    }

    @Test
    public void testToolParse() throws JsonProcessingException {
        String input = "To get the payment status and date of transaction T1005, I will use the retrievePaymentStatus and retrievePaymentDate functions.\n" +
                "\n" +
                "[{\"name\": \"retrievePaymentStatus\", \"arguments\": {\"arg0\": \"T1005\"}}]\n" +
                "[{\"name\": \"retrievePaymentDate\", \"arguments\": {\"arg0\": \"T1005\"}}]\n" +
                "\n" +
                "Please wait while I retrieve the data...";

        List<String> jsonCalls = JsonSupport.extractJsonFromString(input);

        Assert.assertEquals(2, jsonCalls.size());

        List<ToolCall> toolCalls = new ArrayList<>(jsonCalls.size());
        for (String jsonCall : jsonCalls) {
            if (jsonCall.startsWith("[")) {
                List<ToolCall> toolCallList = JsonSupport.om.readValue(jsonCall, new TypeReference<>() {});
                toolCalls.addAll(toolCallList);
            } else {
                ToolCall toolCall = JsonSupport.om.readValue(jsonCall, ToolCall.class);
                toolCalls.add(toolCall);
            }
        }

        // Remove duplicates
        toolCalls = toolCalls.stream().sorted(Comparator.comparing(ToolCall::getName)).distinct().collect(Collectors.toList());

        Assert.assertEquals(2, toolCalls.size());
    }

    @Test
    public void testPromptBuilderSession() {
        String modelPrefix = "../models/Qwen_Qwen2.5-0.5B-Instruct-JQ4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
        PromptSupport.Builder builder = tokenizer.promptSupport().get().builder();
        builder.addSystemMessage("You always respond as a pirate");
        builder.addUserMessage("What is the weather in paris right now?");
        builder.addGenerationPrompt(true);

        Tool t = Tool.from(
            Function.builder()
                .name("get_current_temperature")
                .description("Simulates getting the current temperature at a location.")
                .addParameter("location", "string", "The location to get the temperature for, in the format \"City, Country\".", true)
                .addParameter("unit", "string", "The unit to return the temperature in (e.g., \"celsius\", \"fahrenheit\").", true)
                .build()
        );

        PromptContext prompt = builder.build(t);
        Assert.assertEquals(
                "<|im_start|>system\n" + "You always respond as a pirate\n"
                        + "\n"
                        + "# Tools\n"
                        + "\n"
                        + "You may call one or more functions to assist with the user query.\n"
                        + "\n"
                        + "You are provided with function signatures within <tools></tools> XML tags:\n"
                        + "<tools>\n"
                        + "{\"type\": \"function\", \"function\": {\"name\": \"get_current_temperature\", \"description\": \"Simulates getting the current temperature at a location.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The location to get the temperature for, in the format \\\"City, Country\\\".\"}, \"unit\": {\"type\": \"string\", \"description\": \"The unit to return the temperature in (e.g., \\\"celsius\\\", \\\"fahrenheit\\\").\"}}, \"required\": [\"location\", \"unit\"]}}}\n"
                        + "</tools>\n"
                        + "\n"
                        + "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                        + "<tool_call>\n"
                        + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                        + "</tool_call><|im_end|>\n"
                        + "<|im_start|>user\n"
                        + "What is the weather in paris right now?<|im_end|>\n"
                        + "<|im_start|>assistant\n",
                prompt.getPrompt());

        prompt = tokenizer.promptSupport().get().builder().addUserMessage("This is a test").stripPreamble().build();
        Assert.assertEquals(
                "<|im_start|>user\n" + "This is a test<|im_end|>\n" + "<|im_start|>assistant\n", prompt.getPrompt());
    }
}
