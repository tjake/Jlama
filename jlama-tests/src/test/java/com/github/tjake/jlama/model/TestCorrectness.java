package com.github.tjake.jlama.model;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.gpt2.GPT2Tokenizer;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.safetensors.tokenizer.WordPieceTokenizer;

import com.google.common.io.Resources;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;


public class TestCorrectness {
    private static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, true);

    @Test
    public void testBPETokenizer() {
        String modelPrefix = "../models/Llama-2-7b-chat-hf";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));

        String p =  "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[]{518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580, 29962, 673, 29871};

        Assert.assertArrayEquals(expected, actual);

        System.out.println(tokenizer.decode(actual));

    }

    @Test
    public void TestLLamaTokenizer() throws IOException {
        String modelPrefix = "../models/Llama-2-7b-chat-hf";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));

        String p =  "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[]{518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580, 29962, 673, 29871};

        Assert.assertArrayEquals(expected, actual);

        String out = tokenizer.decode(actual);
        Assert.assertEquals(p, out);

        String s = tokenizer.decode(518);
        Assert.assertEquals(" [", s);

        long[] token = tokenizer.encode(p + "\n");
        expected = new long[]{518, 25580, 29962, 24948, 592, 263, 2958, 446, 29889, 29871, 243, 162, 147, 180, 518, 29914, 25580, 29962, 673, 29871, 13};
        Assert.assertArrayEquals(expected, token);
    }

    @Test
    public void TestRope() throws IOException {
        double[] expected = new double[]{8.4147e-01,  7.6172e-01,  6.8156e-01,  6.0469e-01,  5.3317e-01,
                4.6795e-01,  4.0931e-01,  3.5711e-01,  3.1098e-01,  2.7043e-01,
                2.3492e-01,  2.0391e-01,  1.7689e-01,  1.5338e-01,  1.3296e-01,
                1.1522e-01,  9.9833e-02,  8.6488e-02,  7.4919e-02,  6.4893e-02,
                5.6204e-02,  4.8678e-02,  4.2157e-02,  3.6509e-02,  3.1618e-02,
                2.7381e-02,  2.3712e-02,  2.0534e-02,  1.7782e-02,  1.5399e-02,
                1.3335e-02,  1.1548e-02,  9.9998e-03,  8.6595e-03,  7.4989e-03,
                6.4938e-03,  5.6234e-03,  4.8697e-03,  4.2170e-03,  3.6517e-03,
                3.1623e-03,  2.7384e-03,  2.3714e-03,  2.0535e-03,  1.7783e-03,
                1.5399e-03,  1.3335e-03,  1.1548e-03,  1.0000e-03,  8.6596e-04,
                7.4989e-04,  6.4938e-04,  5.6234e-04,  4.8697e-04,  4.2170e-04,
                3.6517e-04,  3.1623e-04,  2.7384e-04,  2.3714e-04,  2.0535e-04,
                1.7783e-04,  1.5399e-04,  1.3335e-04,  1.1548e-04};

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(128, 4096 * 2, 10000.0 , 1.0);

        for (int i = 0; i < 64; i++)
            Assert.assertEquals(expected[i], ropeFreqs[i + 64][1], 0.0001);


        expected = new double[]{ 0.9200, -0.9031, -0.7639, -0.6592, -0.9904, -0.2474,  0.9597, -0.9819,
                0.9835, -0.9696,  0.5065,  0.5448, -0.9266, -0.4176,  0.7772,  0.8945,
                0.1165, -0.6750, -0.9962, -0.8492, -0.4416,  0.0250,  0.4284,  0.7205,
                0.8991,  0.9835,  0.9986,  0.9673,  0.9078,  0.8336,  0.7536,  0.6736,
                0.5972,  0.5263,  0.4617,  0.4037,  0.3522,  0.3066,  0.2666,  0.2316,
                0.2010,  0.1744,  0.1512,  0.1310,  0.1136,  0.0984,  0.0852,  0.0738,
                0.0640,  0.0554,  0.0480,  0.0415,  0.0360,  0.0312,  0.0270,  0.0234,
                0.0202,  0.0175,  0.0152,  0.0131,  0.0114,  0.0099,  0.0085,  0.0074};


        for (int i = 0; i < 64; i++)
            Assert.assertEquals(expected[i], ropeFreqs[i + (64 * 64)][1], 0.0001);
    }

    @Test
    public void testRope2() throws IOException {
        List<List<Float>> real = om.readerFor(new TypeReference<ArrayList<ArrayList<Float>>>() {}).readValue(Resources.getResource("real.json"));
        List<List<Float>> imag = om.readerFor(new TypeReference<ArrayList<ArrayList<Float>>>() {}).readValue(Resources.getResource("imag.json"));

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(128, 2048, 10000.0, 1.0 );

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
    public void testGptTokenizer() throws IOException
    {
        String modelPrefix = "../models/gpt2-medium";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new GPT2Tokenizer(Paths.get(modelPrefix));

        String p =  "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[]{58, 38604, 60, 14026, 502, 257, 9707, 13, 12520, 238, 109, 46581, 38604, 60, 23998, 220};


        String d = tokenizer.decode(actual);
        System.out.println(d);

        Assert.assertArrayEquals(expected, actual);
        Assert.assertEquals(p, d);
    }

    @Test
    public void testNeoTokenizer() throws IOException
    {
        String modelPrefix = "../models/deepseek-coder-1.3b-base";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        Tokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));

        String p =  "#write a quicksort algorithm";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[]{2, 6449, 245, 3383, 3724, 6713};

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

        String p =  "[INST] Tell me a joke. \uD83D\uDC31 [/INST] Answer ";

        long[] actual = tokenizer.encode(p);
        long[] expected = new long[]{101, 1031, 16021, 2102, 1033, 2425, 2033, 1037, 8257, 1012, 100, 1031, 1013, 16021, 2102, 1033, 3437, 102};


        String decodeExpected = "[CLS] [ inst ] tell me a joke. [UNK] [ / inst ] answer [SEP]";

        String decodeActual = tokenizer.decode(actual);
        System.out.println(decodeActual);

        Assert.assertArrayEquals(expected, actual);
    }
}
