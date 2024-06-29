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

import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import java.util.UUID;
import java.util.function.BiConsumer;

/**
 * Used to define a function that generates tokens from a prompt
 */
public interface Generator {

    enum FinishReason {
        MAX_TOKENS,
        STOP_TOKEN,
    }

    class Response {
        public final String text;
        public final FinishReason finishReason;
        public final int promptTokens;
        public final int generatedTokens;
        public final long promptTimeMs;
        public final long generateTimeMs;

        public Response(String text, FinishReason finishReason, int promptTokens, int generatedTokens, long promptTimeMs, long generateTimeMs) {
            this.text = text;
            this.finishReason = finishReason;
            this.promptTokens = promptTokens;
            this.generatedTokens = generatedTokens;
            this.promptTimeMs = promptTimeMs;
            this.generateTimeMs = generateTimeMs;
        }

        @Override
        public String toString() {
            return "GenerateResponse{" +
                    "text='" + text + '\'' +
                    ", finishReason=" + finishReason +
                    ", promptTokens=" + promptTokens +
                    ", generatedTokens=" + generatedTokens +
                    ", promptTimeMs=" + promptTimeMs +
                    ", generateTimeMs=" + generateTimeMs +
                    '}';
        }
    }

    Response generate(
            UUID session,
            String prompt,
            float temperature,
            int ntokens,
            boolean useEOS,
            BiConsumer<String, Float> onTokenWithTimings);

    Tokenizer getTokenizer();
}
