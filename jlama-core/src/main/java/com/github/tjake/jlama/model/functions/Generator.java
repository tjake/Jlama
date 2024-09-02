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

import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.safetensors.prompt.ToolCall;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.function.BiConsumer;

/**
 * Used to define a function that generates tokens from a prompt
 */
public interface Generator {

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

    Response generate(
        UUID session,
        PromptContext promptContext,
        float temperature,
        int ntokens,
        BiConsumer<String, Float> onTokenWithTimings
    );

    Tokenizer getTokenizer();
}
