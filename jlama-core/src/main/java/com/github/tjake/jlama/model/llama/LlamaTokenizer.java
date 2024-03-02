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
package com.github.tjake.jlama.model.llama;

import com.github.tjake.jlama.safetensors.tokenizer.BPETokenizer;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Collectors;

public class LlamaTokenizer extends BPETokenizer {
    static final String SPIECE_UNDERLINE = "▁";

    private static BiMap<Integer, Integer> alteredBytes; // Codepoint and Token mapping needed for legacy mode

    static {
        // https://github.com/openai/gpt-2/blob/master/src/encoder.py#L19
        alteredBytes = HashBiMap.create();
        int i = 0;
        for (int c = 0; c < 256; c++) {
            if ((c < '!' || c > '~') && (c < '¡' || c > '¬') && (c < '®' || c > 'ÿ')) {
                int codepoint = (i++ + 256);
                alteredBytes.put(c, codepoint);
            }
        }
    }

    private final int byteFallbackEncodingOffset;

    public LlamaTokenizer(Path modelRoot) {
        super(modelRoot);

        this.byteFallbackEncodingOffset = 3;
    }

    @Override
    protected long encodeCharacterAsToken(byte c) {
        return Byte.toUnsignedLong(c) + byteFallbackEncodingOffset;
    }

    @Override
    protected Optional<Character> maybeDecodeTokenAsCharacter(long id) {
        // Handle ascii codes (shifted by 3 in vocab)
        if (model.byteFallback && id >= byteFallbackEncodingOffset && id < 256 + byteFallbackEncodingOffset) {
            char c = (char) (id - byteFallbackEncodingOffset);
            return Optional.of(c);
        }

        return Optional.empty();
    }

    @Override
    protected String preProcess(String sentence) {
        if (!model.isLegacy()) {
            if (!sentence.isEmpty() && !sentence.startsWith(" ")) {
                sentence = " " + sentence;
            }
            return sentence.replace(" ", SPIECE_UNDERLINE);
        } else {
            return sentence.codePoints()
                    .map(c -> alteredBytes.getOrDefault(c, c))
                    .mapToObj(Character::toString)
                    .collect(Collectors.joining());
        }
    }

    @Override
    protected String postProcess(String sentence) {
        return sentence.stripLeading();
    }

    @Override
    protected String postProcessToken(String decoded) {
        if (decoded == null) decoded = model.unkToken;

        decoded = decoded.replaceAll("</?s>", "");
        decoded = decoded.replaceAll(SPIECE_UNDERLINE, " ");

        if (model.isLegacy()) {
            return decoded.codePoints()
                    .map(c -> alteredBytes.inverse().getOrDefault(c, c))
                    .mapToObj(Character::toString)
                    .collect(Collectors.joining());
        }

        return decoded;
    }
}
