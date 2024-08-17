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

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.TensorShape;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.DebugSupport;
import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.github.tjake.jlama.util.DebugSupport.debug;

public abstract class AbstractModel implements Generator {
    private static final Logger logger = LoggerFactory.getLogger(AbstractModel.class);

    public enum InferenceType {
        INPUT_TO_EMBEDDING(true, false, false),
        OUTPUT_TO_TOKEN(false, true, false),
        FORWARD_PASS(true, false, true),
        FULL_GENERATION(true, true, true);

        final boolean isInput;
        final boolean isOutput;
        final boolean isFwdPass;

        InferenceType(boolean isInput, boolean isOutput, boolean isFwdPass) {
            this.isInput = isInput;
            this.isOutput = isOutput;
            this.isFwdPass = isFwdPass;
        }

        public boolean isEmbedding() {
            return isInput;
        }

        public boolean isOutput() {
            return isOutput;
        }

        public boolean isFwdPass() {
            return isFwdPass;
        }
    }

    protected final InferenceType inferenceType;
    protected final Config c;
    protected final WeightLoader weights;
    protected final Tokenizer tokenizer;
    protected final DType modelDType;
    protected final DType workingDType;
    protected final DType workingQType;
    protected final Optional<DType> modelQType;
    protected EmbedInput embedInput;
    protected SampleOutput sampleOutput;
    protected TransformerBlock[] transformerBlocks;
    protected KvBufferCache kvBufferCache;

    protected AbstractModel(
            InferenceType inferenceType,
            Config c,
            WeightLoader w,
            Tokenizer t,
            DType workingMemoryDType,
            DType workingMemoryQType,
            Optional<DType> modelQType) {
        this.inferenceType = inferenceType;
        this.c = c;
        this.weights = w;
        this.tokenizer = t;
        this.modelDType = w.getModelDType();
        this.workingDType = workingMemoryDType;
        this.modelQType = modelQType;
        this.kvBufferCache = new KvBufferCache(this);

        // FIXME: This is a hack to support Avoid Q8F32 evals
        if (modelDType == DType.F32 && workingMemoryQType != DType.F32 && modelQType.isEmpty()) {
            workingMemoryQType = DType.F32;
        }

        // FIXME: This is a hack to support Avoid Q8BF16 evals
        if (modelDType == DType.BF16 && workingMemoryQType != DType.BF16 && modelQType.isEmpty()) {
            workingMemoryQType = DType.BF16;
        }

        if (workingMemoryQType != workingMemoryDType) {
            boolean supportsQType;
            AbstractTensor tmp = makeTensor(Q8ByteBufferTensor.BLOCK_SIZE);
            try (AbstractTensor tmp2 = TensorOperationsProvider.get()
                    .quantize(tmp, workingMemoryQType, 0, Q8ByteBufferTensor.BLOCK_SIZE)) {
                supportsQType = tmp2.dType() == workingMemoryQType;
                if (!supportsQType) {
                    logger.warn(
                            "Quantized memory type {} not supported, falling back to {}",
                            workingMemoryQType,
                            workingMemoryDType);
                    this.workingQType = this.workingDType;
                } else {
                    this.workingQType = workingMemoryQType;
                }
            }
        } else {
            this.workingQType = workingMemoryQType;
        }

        logger.info("Working memory type = {}, Quantized memory type = {}", this.workingDType, this.workingQType);

        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
    }

    protected abstract EmbedInput loadInputWeights();

    protected abstract TransformerBlock[] loadTransformerBlockWeights();

    protected abstract SampleOutput loadOutputWeights();

    public DType getWorkingDType() {
        return workingDType;
    }

    public Config getConfig() {
        return c;
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Optional<PromptSupport> promptSupport() {
        return tokenizer.promptSupport();
    }

    public AbstractTensor makeTensor(int... shape) {
        TensorShape s;
        if (c.offset().isPresent() && shape[shape.length - 1] == c.embeddingLength)
            s = TensorShape.sparse(shape, c.offset().get());
        else s = TensorShape.of(shape);

        return c.tensorCache.get(workingDType, s);
    }

    public AbstractTensor makeFullTensor(int... shape) {
        return c.tensorCache.get(workingDType, TensorShape.of(shape));
    }

    protected AbstractTensor maybeQuantize(AbstractTensor t) {
        AbstractTensor t2 = c.tensorCache.get(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, Ints.checkedCast(t.size()));
        return t2;
    }

    protected AbstractTensor forward(int token_id, int pos, AbstractTensor kvbuf) {
        return forward(token_id, pos, kvbuf, Optional.empty(), Optional.empty());
    }

    /**
     * This is a distributed version of forward pass that serves as a coordination point for the
     * distributed model.  The layers are split into one or more heads and each head is processed
     * by a different node.
     *
     * @param token_id
     * @param pos
     * @param kvbuf
     * @return
     */
    public AbstractTensor forward(
            int token_id,
            int pos,
            AbstractTensor kvbuf,
            Optional<BiFunction<Float, Float, Pair<Float, Float>>> normReducer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        AbstractTensor embedding = embedInput.inputTokenToEmbedding(token_id, pos);

        debug("EMBEDDING TOKEN", token_id);
        debug("TOKEN POSITION", pos);

        for (int i = c.layerStart(); i < c.layerEnd(); i++) {
            AbstractTensor kvlayer = kvbuf.slice(true, i);
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[i].forward(embedding, pos, kvlayer, normReducer, tensorReducer);
            ref.close();
        }

        return embedding;
    }

    protected AbstractTensor batchForwardSlow(int[] token_ids, int startPos, AbstractTensor kvbuf) {
        AbstractTensor last = null;
        for (int i = 0; i < token_ids.length; i++) {
            if (last != null) last.close();
            last = forward(token_ids[i], startPos + i, kvbuf);
        }

        return last;
    }

    protected AbstractTensor batchForward(int[] token_ids, int startPos, AbstractTensor kvbuf) {

        AbstractTensor embedding = embedInput.batchInputsToEmbeddings(token_ids, startPos);
        for (int i = c.layerStart(); i < c.layerEnd(); i++) {
            AbstractTensor kvlayer = kvbuf.slice(true, i);
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[i].forward(embedding, startPos, kvlayer, Optional.empty(), Optional.empty());
            ref.close();
        }

        return embedding;
    }

    public int sample(AbstractTensor output, float temperature, float uniformSample, AbstractTensor logits) {
        try (AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(output)) {
            // This is a mix of argmax and sampling with softmax
            VectorMath.pchunk(0, c.vocabularySize, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get()
                        .dotProductChunk(
                                logits,
                                embedding,
                                sampleOutput.getOutputLogitsWeights(),
                                0,
                                c.embeddingLength,
                                chunkStart,
                                chunkSize);
            });

            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (v > maxv) {
                    maxi = i;
                    maxv = v;
                }
            }

            if (temperature == 0.0) {
                return maxi;
            }

            float sum = 0;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = (float) Math.exp((logits.get(0, i) - maxv) / temperature);
                sum += v;
                logits.set(v, 0, i);
            }

            float acc = 0;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = logits.get(0, i) / sum;
                acc += v;
                if (acc >= uniformSample) return i;
            }

            return c.vocabularySize - 1;
        }
    }

    public Response generate(
            UUID sessionId,
            String prompt,
            float temperature,
            int ntokens,
            boolean useEOS,
            BiConsumer<String, Float> onTokenWithTimings) {
        long[] encoded = tokenizer.encode(prompt);

        // Remove BOS token if it's the first token, we explicitly add it below
        if (encoded.length > 0 && encoded[0] == c.bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length - 1);
        }

        Preconditions.checkArgument(encoded.length < c.contextLength);

        AbstractTensor kvmem = kvBufferCache.getKvBuffer(sessionId); // k and v for context window
        Integer startPos = (Integer) kvmem.getMetadata(KvBufferCache.TOKEN_COUNT); // Number of tokens in the buffer
        if (startPos == null) startPos = 0;

        logger.debug("Starting at token {} for session {}", startPos, sessionId);

        if (ntokens > c.contextLength) ntokens = c.contextLength;

        FinishReason reason = FinishReason.MAX_TOKENS;
        int promptLength;
        long promptBatchTime;
        int tokensGenerated;
        StringBuilder sb = new StringBuilder();

        try (AbstractTensor logits = makeTensor(c.vocabularySize)) {
            int[] promptTokens = new int[useEOS ? (1 + encoded.length + 1) : (1 + encoded.length)];

            promptTokens[0] = c.bosToken;
            for (int i = 1; i <= encoded.length; i++) promptTokens[i] = Ints.checkedCast(encoded[i - 1]);
            promptLength = encoded.length;

            if (useEOS) {
                promptTokens[promptTokens.length - 1] = getConfig().eosTokens.getLast(); // Add EOS
                promptLength++;
            }

            long start = System.currentTimeMillis();
            long promptStart = start;
            // Batch Process Prompt
            AbstractTensor last = DebugSupport.isDebug()
                    ? batchForwardSlow(promptTokens, startPos, kvmem)
                    : batchForward(promptTokens, startPos, kvmem);

            promptBatchTime = System.currentTimeMillis() - start;
            float batchMsPerToken = Math.round((((double) promptBatchTime) / (double) promptLength));
            logger.debug("{} prompt tokens in {}ms | {}ms per token", promptLength, promptBatchTime, batchMsPerToken);

            float genMsPerToken = 0;
            tokensGenerated = 0;
            int next = sample(
                    last.slice(last.shape().first() - 1),
                    temperature,
                    ThreadLocalRandom.current().nextFloat(),
                    logits);
            last.close();
            try {
                String c = tokenizer.decode(next);
                onTokenWithTimings.accept(c, batchMsPerToken);
                sb.append(c);
            } catch (Exception e) {
                logger.error("Failed to decode token {}", next, e);
            }

            start = System.currentTimeMillis();
            for (int i = startPos + promptTokens.length; i < ntokens; i++) {
                AbstractTensor output = forward(next, i, kvmem);
                tokensGenerated++;

                next = sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);

                if (logger.isTraceEnabled()) logger.trace("Sampled token {} with temperature {}", next, temperature);
                output.close();

                kvmem.setMetadata(KvBufferCache.TOKEN_COUNT, i);

                // Model may tell us it's done
                if (c.eosTokens.contains(next)) {
                    reason = FinishReason.STOP_TOKEN;
                    break;
                }

                try {
                    String c = tokenizer.decode(next);
                    genMsPerToken = (System.currentTimeMillis() - start) / (float) (tokensGenerated);
                    onTokenWithTimings.accept(c, genMsPerToken);
                    sb.append(c);
                } catch (Exception e) {
                    logger.error("Failed to decode token {}", next, e);
                }
            }

            long end = System.currentTimeMillis();

            Response response =
                    new Response(sb.toString(), reason, promptLength, tokensGenerated, promptBatchTime, end - start);
            logger.debug(String.format(
                    "\n\nelapsed: %ds, prompt %.1fms per token, gen %.1fms per token\n",
                    TimeUnit.MILLISECONDS.toSeconds(end - promptStart), batchMsPerToken, genMsPerToken));

            return response;
        }
    }
}
