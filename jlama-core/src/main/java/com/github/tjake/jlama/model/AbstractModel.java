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

import static com.github.tjake.jlama.util.DebugSupport.debug;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.github.tjake.jlama.safetensors.prompt.Tool;
import com.github.tjake.jlama.safetensors.prompt.ToolCall;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.TensorShape;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.DebugSupport;
import com.github.tjake.jlama.util.JsonSupport;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        Optional<DType> modelQType
    ) {
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
            AbstractTensor tmp = makeDenseTensor(Q8ByteBufferTensor.BLOCK_SIZE);
            try (AbstractTensor tmp2 = TensorOperationsProvider.get().quantize(tmp, workingMemoryQType, 0, Q8ByteBufferTensor.BLOCK_SIZE)) {
                supportsQType = tmp2.dType() == workingMemoryQType;
                if (!supportsQType) {
                    logger.warn("Quantized memory type {} not supported, falling back to {}", workingMemoryQType, workingMemoryDType);
                    this.workingQType = this.workingDType;
                } else {
                    this.workingQType = workingMemoryQType;
                }
            }
        } else {
            this.workingQType = workingMemoryQType;
        }

        logger.info("Model type = {}, Working memory type = {}, Quantized memory type = {}", this.modelDType, this.workingDType, this.workingQType);

        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
    }

    protected abstract EmbedInput loadInputWeights();

    protected abstract TransformerBlock[] loadTransformerBlockWeights();

    protected abstract SampleOutput loadOutputWeights();

    public abstract ModelSupport.ModelType getModelType();

    public InferenceType getInferenceType() {
        return inferenceType;
    }

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
        TensorShape s = TensorShape.of(shape);
        return c.tensorCache.get(workingDType, s);
    }

    public AbstractTensor makeDenseTensor(int... shape) {
        return c.tensorCache.get(workingDType, TensorShape.of(shape));
    }

    protected AbstractTensor maybeQuantize(AbstractTensor t) {
        AbstractTensor t2 = c.tensorCache.get(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, Ints.checkedCast(t.size()));
        return t2;
    }

    protected AbstractTensor forward(int token_id, int pos, KvBufferCache.KvBuffer kvbuf) {
        return forward(token_id, pos, kvbuf, Optional.empty());
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
        KvBufferCache.KvBuffer kvbuf,
        Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        AbstractTensor embedding = embedInput.inputTokenToEmbedding(token_id, pos);

        debug("EMBEDDING TOKEN", token_id);
        debug("TOKEN POSITION", pos);

        for (int i = c.dctx().layerStart; i < c.dctx().layerEnd; i++) {
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[i].forward(embedding, pos, kvbuf, tensorReducer);
            ref.close();
        }

        return embedding;
    }

    protected AbstractTensor batchForwardSlow(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf) {
        AbstractTensor last = null;
        for (int i = 0; i < token_ids.length; i++) {
            if (last != null) last.close();
            last = forward(token_ids[i], startPos + i, kvbuf);
        }

        return last;
    }

    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf) {
        return batchForward(token_ids, startPos, kvbuf, Optional.empty());
    }

    public AbstractTensor batchForward(
        int[] token_ids,
        int startPos,
        KvBufferCache.KvBuffer kvbuf,
        Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        AbstractTensor embedding = embedInput.batchInputsToEmbeddings(token_ids, startPos);
        for (int i = c.dctx().layerStart; i < c.dctx().layerEnd; i++) {
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[i].forward(embedding, startPos, kvbuf, tensorReducer);
            ref.close();
        }

        return embedding;
    }

    @Override
    public float[] embed(String input) {
        int[] encoded = Arrays.stream(tokenizer.encode(input)).mapToInt(Ints::checkedCast).toArray();
        Preconditions.checkArgument(encoded.length < c.contextLength);
        float[] outputEmbedding = new float[c.embeddingLength];

        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(UUID.randomUUID())) {
            int promptLength = encoded.length;
            float avgp = 1.0f / promptLength;

            AbstractTensor r = batchForward(encoded, 0, kvmem);
            for (int i = 0; i < promptLength; i++) {
                AbstractTensor output = r.slice(i);

                // Average Pooling
                for (int ii = 0; ii < c.embeddingLength; ii++)
                    outputEmbedding[ii] += output.get(0, ii) * avgp;
            }
            r.close();
            VectorMath.l2normalize(outputEmbedding);
        }
        return outputEmbedding;
    }

    public int sample(AbstractTensor output, float temperature, float uniformSample, AbstractTensor logits) {
        try (AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(output)) {
            // This is a mix of argmax and sampling with softmax
            VectorMath.pchunk(0, c.vocabularySize, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(logits, embedding, sampleOutput.getOutputLogitsWeights(), 0, c.embeddingLength, chunkStart, chunkSize);
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



    @Override
    public Response generate(
        UUID sessionId,
        PromptContext promptContext,
        float temperature,
        int ntokens,
        BiConsumer<String, Float> onTokenWithTimings
    ) {
        long[] encoded = tokenizer.encode(promptContext.getPrompt());

        // Remove BOS token if it's the first token, we explicitly add it below
        if (encoded.length > 0 && encoded[0] == c.bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }

        Preconditions.checkArgument(encoded.length < c.contextLength && encoded.length < ntokens, "Prompt exceeds max tokens");

        KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(sessionId); // k and v for context window
        int startPos = kvmem.getCurrentContextPosition(); // Number of tokens in the buffer

        logger.debug("Starting at token {} for session {}", startPos, sessionId);

        if (ntokens > c.contextLength) ntokens = c.contextLength;

        FinishReason reason = FinishReason.MAX_TOKENS;
        int promptLength;
        long promptBatchTime;
        int tokensGenerated;
        StringBuilder responseText = new StringBuilder();
        StringBuilder responseTextWithSpecialTokens = new StringBuilder();

        try (AbstractTensor logits = makeDenseTensor(c.vocabularySize)) {
            int[] promptTokens = new int[(1 + encoded.length)];

            promptTokens[0] = c.bosToken;
            for (int i = 1; i <= encoded.length; i++)
                promptTokens[i] = Ints.checkedCast(encoded[i - 1]);
            promptLength = encoded.length;

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
            int next = sample(last.slice(last.shape().first() - 1), temperature, ThreadLocalRandom.current().nextFloat(), logits);
            last.close();
            try {
                String c = tokenizer.decode(next);
                if (tokenizer.getModel().isSpecialToken(next)) {
                    responseTextWithSpecialTokens.append(c);
                } else {
                    onTokenWithTimings.accept(c, batchMsPerToken);
                    responseText.append(c);
                    responseTextWithSpecialTokens.append(c);
                }
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

                kvmem.incrementContextPosition();

                // Model may tell us it's done
                if (c.eosTokens.contains(next)) {
                    reason = FinishReason.STOP_TOKEN;
                    break;
                }

                try {
                    String c = tokenizer.decode(next);

                    if (tokenizer.getModel().isSpecialToken(next)) {
                        responseTextWithSpecialTokens.append(c);
                    } else {
                        genMsPerToken = (System.currentTimeMillis() - start) / (float) (tokensGenerated);
                        onTokenWithTimings.accept(c, genMsPerToken);
                        responseTextWithSpecialTokens.append(c);
                        responseText.append(c);
                    }
                } catch (Exception e) {
                    logger.error("Failed to decode token {}", next, e);
                }
            }

            long end = System.currentTimeMillis();

            Response response = new Response(
                responseText.toString(),
                responseTextWithSpecialTokens.toString(),
                reason,
                promptLength,
                tokensGenerated,
                promptBatchTime,
                end - start
            );
            logger.debug(
                String.format(
                    "\n\nelapsed: %ds, prompt %.1fms per token, gen %.1fms per token\n",
                    TimeUnit.MILLISECONDS.toSeconds(end - promptStart),
                    batchMsPerToken,
                    genMsPerToken
                )
            );

            return postProcessResponse(promptContext, response);
        }
    }

    /**
     * This is a hook for subclasses to post process the response before returning it to the caller.
     * For example this can be used to handle tool calls.
     * @param response
     * @return */
    protected Generator.Response postProcessResponse(PromptContext promptContext, Generator.Response response) {
        if (!tokenizer.getModel().hasToolSupport() || !promptContext.hasTools() || response.finishReason != FinishReason.STOP_TOKEN) {
            return response;
        }

        // Look for a tool call based on the tool names
        List<Tool> tools = promptContext.getTools().get();
        boolean foundTool = false;
        for (Tool tool : tools) {
            if (response.responseTextWithSpecialTokens.contains(tool.getFunction().getName())) {
                foundTool = true;
                break;
            }
        }

        if (!foundTool) {
            return response;
        }

        try {
            // If we found a tool call, we need to extract the tool call from the response
            List<String> jsonCalls = JsonSupport.extractJsonFromString(response.responseText);
            if (jsonCalls.isEmpty()) {
                logger.warn("Tool call detected but no tool call found in response: {}", response.responseText);
                return response;
            }

            logger.debug("Found tool calls: {}", jsonCalls);
            List<ToolCall> toolCalls = new ArrayList<>(jsonCalls.size());
            for (String jsonCall : jsonCalls) {
                if (jsonCall.startsWith("[")) {
                    List<ToolCall> toolCallList = JsonSupport.om.readValue(jsonCall, ToolCall.toolCallListTypeReference);
                    toolCalls.addAll(toolCallList);
                } else {
                    ToolCall toolCall = JsonSupport.om.readValue(jsonCall, ToolCall.class);
                    toolCalls.add(toolCall);
                }
            }

            // Remove duplicates
            toolCalls = toolCalls.stream().sorted(Comparator.comparing(ToolCall::getName)).distinct().collect(Collectors.toList());

            for (int i = 0; i < toolCalls.size(); i++) {
                // Standard is to use 9 digit ids
                toolCalls.get(i).setId(String.format("%09d", i));
            }

            return response.copyWithToolCalls(toolCalls);
        } catch (JsonProcessingException e) {
            logger.error("Failed to parse tool call from response: {}", response.responseText, e);
        }

        return response;
    }
}
