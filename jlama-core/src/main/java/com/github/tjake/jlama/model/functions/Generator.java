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
package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.github.tjake.jlama.safetensors.prompt.ToolCall;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

import java.io.Closeable;
import java.util.*;
import java.util.function.BiConsumer;

/**
 * Used to define a function that generates tokens from a prompt
 */
public interface Generator extends Closeable {

    enum FinishReason {
        MAX_TOKENS,
        STOP_TOKEN,
        TOOL_CALL,
        ERROR
    }

    class Response {
        public final String responseText;
        public final String responseTextWithSpecialTokens;
        public final FinishReason finishReason;
        public final int promptTokens;
        public final int generatedTokens;
        public final long promptTimeMs;
        public final long generateTimeMs;
        public final List<ToolCall> toolCalls;

        public Response(
            String responseText,
            String responseTextWithSpecialTokens,
            FinishReason finishReason,
            int promptTokens,
            int generatedTokens,
            long promptTimeMs,
            long generateTimeMs
        ) {
            this.responseText = responseText;
            this.responseTextWithSpecialTokens = responseTextWithSpecialTokens;
            this.finishReason = finishReason;
            this.promptTokens = promptTokens;
            this.generatedTokens = generatedTokens;
            this.promptTimeMs = promptTimeMs;
            this.generateTimeMs = generateTimeMs;
            this.toolCalls = Collections.emptyList();
        }

        private Response(
            String responseText,
            String responseTextWithSpecialTokens,
            FinishReason finishReason,
            int promptTokens,
            int generatedTokens,
            long promptTimeMs,
            long generateTimeMs,
            List<ToolCall> toolCalls
        ) {
            this.responseText = responseText;
            this.responseTextWithSpecialTokens = responseTextWithSpecialTokens;
            this.finishReason = finishReason;
            this.promptTokens = promptTokens;
            this.generatedTokens = generatedTokens;
            this.promptTimeMs = promptTimeMs;
            this.generateTimeMs = generateTimeMs;
            this.toolCalls = toolCalls;
        }

        public Response copyWithToolCalls(List<ToolCall> toolCalls) {
            return new Response(
                responseText,
                responseTextWithSpecialTokens,
                FinishReason.TOOL_CALL,
                promptTokens,
                generatedTokens,
                promptTimeMs,
                generateTimeMs,
                toolCalls
            );
        }

        @Override
        public String toString() {
            return "Response{"
                + "responseText='"
                + responseText
                + '\''
                + ", responseTextWithSpecialTokens='"
                + responseTextWithSpecialTokens
                + '\''
                + ", finishReason="
                + finishReason
                + ", promptTokens="
                + promptTokens
                + ", generatedTokens="
                + generatedTokens
                + ", promptTimeMs="
                + promptTimeMs
                + ", generateTimeMs="
                + generateTimeMs
                + '}';
        }
    }

    /**
     * Generate tokens from a prompt
     *
     * @param session the session id
     * @param promptContext the prompt context
     * @param temperature the temperature [0.0, 1.0]
     * @param ntokens the number of tokens to generate
     * @param onTokenWithTimings a callback for each token generated
     * @return the response
     */
    Response generate(
        UUID session,
        PromptContext promptContext,
        float temperature,
        int ntokens,
        BiConsumer<String, Float> onTokenWithTimings
    );

    default Response generate(
            UUID session,
            PromptContext promptContext,
            float temperature,
            int ntokens
    ) {
        return generate(session, promptContext, temperature, ntokens, (s, aFloat) -> {});
    }

    default GenerateBuilder generateBuilder() {
        return new GenerateBuilder(this);
    }

    enum PoolingType {
        MODEL, // Use the model's pooling layers
        AVG,
        MAX,
        SUM
    }

    /**
     * Embed a string
     *
     * @param input the input string
     * @return the embeddings
     */
    float[] embed(String input, PoolingType poolingType);

    /**
     * Classify a string
     *
     * @param input the input string
     * @return the classification (if supported)
     */
    default Map<String, Float> classify(String input, PoolingType poolingType) {
        throw new UnsupportedOperationException("Classification not supported by this model");
    }

    Config getConfig();

    Tokenizer getTokenizer();

    Optional<PromptSupport> promptSupport();

    class GenerateBuilder {
        private UUID session = UUID.randomUUID();
        private PromptContext promptContext;
        private float temperature = 0.0f;
        private int ntokens = 256;
        private BiConsumer<String, Float> onTokenWithTimings = (s, aFloat) -> {};
        private final Generator generator;

        public GenerateBuilder(Generator generator) {
            this.generator = generator;
        }

        public GenerateBuilder session(UUID session) {
            this.session = session;

            return this;
        }

        public GenerateBuilder session(String session) {
            this.session = UUID.fromString(session);

            return this;
        }

        public GenerateBuilder promptContext(PromptContext promptContext) {
            this.promptContext = promptContext;

            return this;
        }

        public GenerateBuilder prompt(String prompt) {
            this.promptContext = PromptContext.of(prompt);

            return this;
        }

        public GenerateBuilder temperature(float temperature) {
            this.temperature = temperature;

            return this;
        }

        public GenerateBuilder ntokens(int ntokens) {
            this.ntokens = ntokens;

            return this;
        }

        public GenerateBuilder onTokenWithTimings(BiConsumer<String, Float> onTokenWithTimings) {
            this.onTokenWithTimings = onTokenWithTimings;

            return this;
        }

        public Response generate() {
            if (promptContext == null) {
                throw new IllegalArgumentException("promptContext cannot be null");
            }

            return generator.generate(session, promptContext, temperature, ntokens, onTokenWithTimings);
        }
    }
}
