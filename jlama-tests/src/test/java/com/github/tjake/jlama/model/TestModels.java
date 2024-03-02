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
import com.github.tjake.jlama.model.bert.BertConfig;
import com.github.tjake.jlama.model.bert.BertModel;
import com.github.tjake.jlama.model.bert.BertTokenizer;
import com.github.tjake.jlama.model.gemma.GemmaConfig;
import com.github.tjake.jlama.model.gemma.GemmaModel;
import com.github.tjake.jlama.model.gemma.GemmaTokenizer;
import com.github.tjake.jlama.model.gpt2.GPT2Config;
import com.github.tjake.jlama.model.gpt2.GPT2Model;
import com.github.tjake.jlama.model.gpt2.GPT2Tokenizer;
import com.github.tjake.jlama.model.llama.LlamaConfig;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.model.mixtral.MixtralConfig;
import com.github.tjake.jlama.model.mixtral.MixtralModel;
import com.github.tjake.jlama.safetensors.*;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.Pair;
import com.google.common.primitives.Ints;
import com.google.common.util.concurrent.AtomicDouble;
import com.google.common.util.concurrent.Uninterruptibles;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import org.jctools.queues.MpmcArrayQueue;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestModels {

    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    private static final Logger logger = LoggerFactory.getLogger(TestModels.class);

    @Test
    public void GPT2Run() throws IOException {
        String modelPrefix = "../models/gpt2-medium";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (RandomAccessFile sc = new RandomAccessFile(modelPrefix + "/model.safetensors", "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights v = SafeTensorSupport.readWeights(bb);
            Tokenizer tokenizer = new GPT2Tokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), GPT2Config.class);
            GPT2Model gpt2 = new GPT2Model(c, v, tokenizer, DType.F32, DType.F32, Optional.of(DType.F32));

            String prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
                    + "previously unexplored valley, in the Andes Mountains. "
                    + "Even more surprising to the researchers was the fact that the unicorns spoke perfect English.";
            gpt2.generate(UUID.randomUUID(), prompt, 0.8f, 256, false, makeOutHandler());
        }
    }

    @Test
    public void LlamaRun() throws Exception {
        String modelPrefix = "../models/Llama-2-7b-chat-hf-jlama-Q4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights =
                SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());
            String prompt = "Simply put, the theory of relativity states that";
            model.generate(UUID.randomUUID(), prompt, 0.7f, 256, false, makeOutHandler());
        }
    }

    @Test
    public void DeepCoderRun() throws Exception {
        String modelPrefix = "../models/deepseek-coder-1.3b-base";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights =
                SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.F32, Optional.empty());
            String prompt = "#write a quicksort algorithm in python";
            model.generate(UUID.randomUUID(), prompt, 0.7f, 256, false, makeOutHandler());
        }
    }

    @Test
    public void MistralRun() throws Exception {
        String modelPrefix = "../models/Mistral-7B-v0.1-jlama-Q4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights =
                SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());
            String prompt = "Simply put, the theory of relativity states that";
            model.generate(UUID.randomUUID(), prompt, 0.7f, 256, false, makeOutHandler());
        }
    }

    @Test
    public void MixtralRun() throws Exception {
        String modelPrefix = "../models/Mixtral-8x7B-Instruct-v0.1-jlama-Q4";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights =
                SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            MixtralConfig c = om.readValue(new File(modelPrefix + "/config.json"), MixtralConfig.class);
            MixtralModel model = new MixtralModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());
            String prompt = "Simply put, the theory of relativity states that";
            model.generate(UUID.randomUUID(), prompt, 0.7f, 256, false, makeOutHandler());
        }
    }

    @Test
    public void GemmaRun() throws Exception {
        String modelPrefix = "../models/gemma-2b";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));
        try (WeightLoader weights =
                SafeTensorSupport.loadWeights(Path.of(modelPrefix).toFile())) {
            GemmaTokenizer tokenizer = new GemmaTokenizer(Paths.get(modelPrefix));
            GemmaConfig c = om.readValue(new File(modelPrefix + "/config.json"), GemmaConfig.class);
            GemmaModel model = new GemmaModel(c, weights, tokenizer, DType.F32, DType.F32, Optional.empty());
            String prompt = "Simply put, the theory of relativity states that";
            model.generate(UUID.randomUUID(), prompt, 0.9f, 256, false, makeOutHandler());
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
                    new String[] {
                        "model.embed_tokens.weight", "lm_head.weight",
                    },
                    null,
                    Optional.of(tmpOut));

            Assert.assertEquals(tmpOut, out);

            WeightLoader weights = SafeTensorSupport.loadWeights(tmpOut.toFile());
            LlamaTokenizer tokenizer = new LlamaTokenizer(tmpOut);
            Config c = om.readValue(new File(tmpOut + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.I8, Optional.empty());

            String prompt = "Lily picked up a flower and gave it to";
            model.generate(UUID.randomUUID(), prompt, 0.7f, 128, false, makeOutHandler());
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
        String modelPrefix = "models/TinyLLama";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        try (RandomAccessFile sc = new RandomAccessFile(modelPrefix + "/model.safetensors", "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights weights = SafeTensorSupport.readWeights(bb);
            LlamaTokenizer tokenizer = new LlamaTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), LlamaConfig.class);
            LlamaModel model = new LlamaModel(c, weights, tokenizer, DType.F32, DType.F32, Optional.of(DType.F32));

            String prompt = "Lily picked up a flower and gave it to";
            model.generate(UUID.randomUUID(), prompt, 0.7f, 128, false, makeOutHandler());
        }
    }

    @Test
    public void BertRun() throws Exception {
        String modelPrefix = "models/e5-small-v2";
        Assume.assumeTrue(Files.exists(Paths.get(modelPrefix)));

        try (RandomAccessFile sc = new RandomAccessFile(modelPrefix + "/model.safetensors", "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights weights = SafeTensorSupport.readWeights(bb);
            Tokenizer tokenizer = new BertTokenizer(Paths.get(modelPrefix));
            Config c = om.readValue(new File(modelPrefix + "/config.json"), BertConfig.class);
            BertModel model = new BertModel(c, weights, tokenizer, DType.F32, DType.F32, Optional.of(DType.F32));

            String base = "A man is eating food.";
            String[] examples = new String[] {
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
            logger.info("took {} seconds, {}ms per emb", elapsed / 1000f, elapsed / 1000f);
        }
    }

    private BiConsumer<String, Float> makeOutHandler() {
        PrintWriter out;
        BiConsumer<String, Float> outCallback;
        if (System.console() == null) {
            AtomicInteger i = new AtomicInteger(0);
            StringBuilder b = new StringBuilder();
            out = new PrintWriter(System.out);
            outCallback = (w, t) -> {
                b.append(w);
                out.println(String.format("%d: %s [took %.2fms])", i.getAndIncrement(), b, t));
                out.flush();
            };
        } else {
            out = System.console().writer();
            outCallback = (w, t) -> {
                out.print(w);
                out.flush();
            };
        }

        return outCallback;
    }

    @Test
    public void testDistibuted() {
        Path model = Paths.get("../models/tiny-random-llama-2");
        Assume.assumeTrue(Files.exists(model));

        AbstractModel mFull =
                ModelSupport.loadModel(model.toFile(), null, DType.F32, DType.F32, Optional.empty(), Optional.empty());

        int len = mFull.getConfig().embeddingLength;
        AbstractModel mFirstHalf = ModelSupport.loadModel(
                AbstractModel.InferenceType.FULL_GENERATION,
                model.toFile(),
                null,
                DType.F32,
                DType.F32,
                Optional.empty(),
                Optional.empty(),
                Optional.of(Pair.create(0, len / 2)));
        AbstractModel mSecondHalf = ModelSupport.loadModel(
                AbstractModel.InferenceType.FULL_GENERATION,
                model.toFile(),
                null,
                DType.F32,
                DType.F32,
                Optional.empty(),
                Optional.empty(),
                Optional.of(Pair.create(len / 2, len / 2)));

        int embLen = mFull.c.embeddingLength;

        // Main one
        AbstractTensor kvmem0 = mFull.makeTensor(
                mFull.c.getNumberOfLayers(), 10, 2, mFull.c.embeddingLength); // k and v are last 2 dims
        AbstractTensor t0 = mFull.embedInput.inputTokenToEmbedding(mFull.c.bosToken, 0);
        AbstractTensor f0 = mFull.transformerBlocks[0].preAttentionNorm.get().forward(t0);

        AbstractTensor q0 = mFull.makeTensor(embLen);
        VectorMath.pchunk(0, embLen, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                    .dotProductChunk(
                            q0,
                            t0,
                            mFull.transformerBlocks[0].attention.queryAttnWeights,
                            0,
                            embLen,
                            chunkStart,
                            chunkLength);
        });

        // Test a double dot
        AbstractTensor qq0 = mFull.makeTensor(embLen);
        VectorMath.pchunk(0, embLen, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                    .dotProductChunk(
                            qq0,
                            q0,
                            mFull.transformerBlocks[0].attention.queryAttnWeights,
                            0,
                            embLen,
                            chunkStart,
                            chunkLength);
        });

        AbstractTensor a0 = mFull.transformerBlocks[0].attention.forward(f0, 0, kvmem0.slice(0), Optional.empty());

        // Two halves
        AtomicDouble sum0 = new AtomicDouble(0);
        CountDownLatch norm0Latch = new CountDownLatch(2);

        BiFunction<Float, Float, Pair<Float, Float>> reducer = (a, b) -> {
            sum0.addAndGet(a);
            norm0Latch.countDown();
            Uninterruptibles.awaitUninterruptibly(norm0Latch);

            return Pair.create(sum0.floatValue(), 0f);
        };

        MpmcArrayQueue<AbstractTensor> sumt = new MpmcArrayQueue<>(2);
        CountDownLatch tensor0Latch = new CountDownLatch(2);

        BiFunction<AbstractTensor, AbstractTensor, Void> treducer = (a, b) -> {
            sumt.offer(a);
            tensor0Latch.countDown();
            Uninterruptibles.awaitUninterruptibly(tensor0Latch);

            AbstractTensor result = a.copyShape();
            for (AbstractTensor t : sumt)
                TensorOperationsProvider.get().accumulate(result, t, 0, Ints.checkedCast(result.size()));

            b.copyFrom(result, 0, 0, Ints.checkedCast(result.size()));
            return null;
        };

        AbstractTensor kvmem1 = mFirstHalf.makeTensor(
                mFull.c.getNumberOfLayers(), 10, 2, mFull.c.embeddingLength); // k and v are last 2 dims
        AbstractTensor t1 = mFirstHalf.embedInput.inputTokenToEmbedding(mFull.c.bosToken, 0);

        AbstractTensor kvmem2 = mSecondHalf.makeTensor(
                mFull.c.getNumberOfLayers(), 10, 2, mFull.c.embeddingLength); // k and v are last 2 dims
        AbstractTensor t2 = mSecondHalf.embedInput.inputTokenToEmbedding(mFull.c.bosToken, 0);

        CompletableFuture<AbstractTensor> f1c = CompletableFuture.supplyAsync(
                () -> mFirstHalf.transformerBlocks[0].preAttentionNorm.get().forward(t1, Optional.of(reducer)));
        CompletableFuture<AbstractTensor> f2c = CompletableFuture.supplyAsync(
                () -> mSecondHalf.transformerBlocks[0].preAttentionNorm.get().forward(t2, Optional.of(reducer)));

        CompletableFuture.allOf(f1c, f2c);
        AbstractTensor f1 = f1c.join();
        AbstractTensor f2 = f2c.join();

        AbstractTensor q1 = mFull.makeTensor(embLen);
        AbstractTensor qq1 = mFull.makeTensor(embLen);

        int off0 = mFirstHalf.c.embeddingSegmentStart();
        int len0 = mFirstHalf.c.embeddingSegmentLength();

        VectorMath.pchunk(0, embLen, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                    .dotProductChunk(
                            q1,
                            t1,
                            mFirstHalf.transformerBlocks[0].attention.queryAttnWeights,
                            off0,
                            len0,
                            chunkStart,
                            chunkLength);
        });

        VectorMath.pchunk(0, embLen, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                    .dotProductChunk(
                            qq1,
                            q1,
                            mFirstHalf.transformerBlocks[0].attention.queryAttnWeights,
                            off0,
                            len0,
                            chunkStart,
                            chunkLength);
        });

        AbstractTensor q2 = mFull.makeTensor(embLen);
        AbstractTensor qq2 = mFull.makeTensor(embLen);

        int off1 = mSecondHalf.c.embeddingSegmentStart();
        int len1 = mSecondHalf.c.embeddingSegmentLength();

        VectorMath.pchunk(0, embLen, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                    .dotProductChunk(
                            q2,
                            t2,
                            mSecondHalf.transformerBlocks[0].attention.queryAttnWeights,
                            off1,
                            len1,
                            chunkStart,
                            chunkLength);
        });
        VectorMath.pchunk(0, embLen, (chunkStart, chunkLength) -> {
            TensorOperationsProvider.get()
                    .dotProductChunk(
                            qq2,
                            q2,
                            mSecondHalf.transformerBlocks[0].attention.queryAttnWeights,
                            off1,
                            len1,
                            chunkStart,
                            chunkLength);
        });

        AbstractTensor a1 = mFirstHalf.transformerBlocks[0].attention.forward(f1, 0, kvmem1.slice(0), Optional.empty());
        AbstractTensor a2 =
                mSecondHalf.transformerBlocks[0].attention.forward(f2, 0, kvmem2.slice(0), Optional.empty());

        AbstractTensor tc = mFull.makeTensor(mFull.c.embeddingLength);
        tc.copyFrom(t1, 0, 0, (int) t1.size());
        tc.copyFrom(t2, 0, (int) t2.size(), (int) t2.size());
        Assert.assertTrue(tensorEquals(tc, t0));

        AbstractTensor fc = mFull.makeTensor(mFull.c.embeddingLength);
        fc.copyFrom(f1, 0, 0, (int) f1.size());
        fc.copyFrom(f2, 0, (int) f2.size(), (int) f2.size());
        Assert.assertTrue(tensorEquals(fc, f0));

        AbstractTensor qc = mFull.makeTensor(mFull.c.embeddingLength);
        qc.copyFrom(q1, 0, 0, (int) q1.size());
        TensorOperationsProvider.get().accumulate(qc, q2, 0, (int) qc.size());
        Assert.assertTrue(tensorEquals(qc, q0));
    }

    static boolean tensorEquals(AbstractTensor a, AbstractTensor b) {
        if (a.size() != b.size()) return false;

        for (int i = 0; i < a.size(); i++) Assert.assertEquals("Position " + i, a.get(i), b.get(i), 0.0000001f);

        return true;
    }
}
