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
package com.github.tjake.jlama.model.qwen2;

import com.github.tjake.jlama.safetensors.tokenizer.BPETokenizer;

import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Collectors;

public class Qwen2Tokenizer extends BPETokenizer {

    public Qwen2Tokenizer(Path modelRoot) {
        super(modelRoot);
    }

    @Override
    protected String preProcess(String sentence) {
        if (model.normalizer() != null) sentence = model.normalizer().normalize(sentence);

        if (model.isLegacy() && !model.byteFallback) {
            sentence = sentence.codePoints()
                .map(c -> alteredBytes.getOrDefault(c, c))
                .mapToObj(Character::toString)
                .collect(Collectors.joining());
        }

        return sentence;
    }

    @Override
    protected long encodeCharacterAsToken(byte c) {
        return Byte.toUnsignedLong(c);
    }

    @Override
    protected Optional<Character> maybeDecodeTokenAsCharacter(long id) {
        return Optional.empty();
    }

    @Override
    protected String postProcessToken(String decoded) {
        if (decoded == null) decoded = model.unkToken;

        if (model.isLegacy() && !model.byteFallback) {
            decoded = decoded.codePoints()
                .map(c -> alteredBytes.inverse().getOrDefault(c, c))
                .mapToObj(Character::toString)
                .collect(Collectors.joining());
        }

        return decoded;
    }
}
