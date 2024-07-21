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
package com.github.tjake.jlama.model.gemma;

import com.github.tjake.jlama.safetensors.tokenizer.BPETokenizer;
import java.nio.file.Path;
import java.util.Optional;

public class GemmaTokenizer extends BPETokenizer {
    static final String SPIECE_UNDERLINE = "▁";

    private final int byteFallbackEncodingOffset;

    public GemmaTokenizer(Path modelRoot) {
        super(modelRoot);

        this.byteFallbackEncodingOffset = 217;
    }

    @Override
    protected long encodeCharacterAsToken(byte c) {
        return Byte.toUnsignedLong(c) + byteFallbackEncodingOffset;
    }

    @Override
    protected Optional<Character> maybeDecodeTokenAsCharacter(long id) {
        // Handle ascii codes (shifted in vocab)
        if (model.byteFallback && id >= byteFallbackEncodingOffset && id < 256 + byteFallbackEncodingOffset) {
            char c = (char) (id - byteFallbackEncodingOffset);
            return Optional.of(c);
        }

        return Optional.empty();
    }

    @Override
    protected String preProcess(String sentence) {
        sentence = sentence.replace(" ", SPIECE_UNDERLINE);

        return sentence;
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

        return decoded;
    }
}
