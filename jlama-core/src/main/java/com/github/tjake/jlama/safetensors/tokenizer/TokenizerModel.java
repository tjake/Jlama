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
package com.github.tjake.jlama.safetensors.tokenizer;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tokenizer model, loosely based on Huggingface's Tokenizer format
 */
public class TokenizerModel {
    @JsonProperty("type")
    public final String type;

    @JsonProperty("unk_token")
    public final String unkToken;

    @JsonProperty("fuse_unk")
    public final boolean fuseUnk;

    @JsonProperty("byte_fallback")
    public final boolean byteFallback;

    @JsonProperty("vocab")
    public final BiMap<String, Long> vocabLookup;

    private PreTokenizer preTokenizer;

    // This is pretty much a hack to support the legacy tokenizer
    private boolean legacy = false;

    @JsonCreator
    public TokenizerModel(
            @JsonProperty("type") String type,
            @JsonProperty("unk_token") String unkToken,
            @JsonProperty("fuse_unk") boolean fuseUnk,
            @JsonProperty("byte_fallback") boolean byteFallback,
            @JsonProperty("vocab") Map<String, Long> vocabLookup) {
        this.type = type;
        this.unkToken = unkToken;
        this.fuseUnk = fuseUnk;
        this.byteFallback = byteFallback;
        this.vocabLookup = ImmutableBiMap.copyOf(vocabLookup);
    }

    public PreTokenizer preTokenizer() {
        return preTokenizer;
    }

    public void setPreTokenizer(PreTokenizer preTokenizer) {
        this.preTokenizer = preTokenizer;
    }

    public boolean isLegacy() {
        return legacy;
    }

    public void setLegacy(boolean legacy) {
        this.legacy = legacy;
    }

    // PreTokenizer class
    public static class PreTokenizer {
        public final String type;

        public final List<PretokenizerItem> pretokenizers;

        @JsonCreator
        public PreTokenizer(
                @JsonProperty("type") String type,
                @JsonProperty("pretokenizers") List<PretokenizerItem> pretokenizers) {
            this.type = type;
            this.pretokenizers = pretokenizers == null ? Collections.emptyList() : ImmutableList.copyOf(pretokenizers);
        }

        public List<String> pretokenize(String sentence) {
            if (pretokenizers.isEmpty()) return Collections.singletonList(sentence);

            Preconditions.checkArgument(type.equalsIgnoreCase("Sequence"), "Invalid pre-tokenizer type: " + type);
            List<String> pieces = List.of(sentence);
            List<String> tmp = new ArrayList<>();
            for (PretokenizerItem item : pretokenizers) {
                for (String piece : pieces) {
                    tmp.addAll(item.pretokenize(piece));
                }

                pieces = tmp;
                tmp = new ArrayList<>();
            }

            return pieces;
        }
    }

    // PretokenizerItem class
    public static class PretokenizerItem {

        public final String type;
        public final Pattern pattern;
        public final String behavior;
        public final Boolean invert;
        public final Boolean individual_digits;
        public final Boolean add_prefix_space;
        public final Boolean trim_offsets;
        public final Boolean use_regex;

        @JsonCreator
        public PretokenizerItem(
                @JsonProperty("type") String type,
                @JsonProperty("pattern") Pattern pattern,
                @JsonProperty("behavior") String behavior,
                @JsonProperty("invert") Boolean invert,
                @JsonProperty("individual_digits") Boolean individual_digits,
                @JsonProperty("add_prefix_space") Boolean add_prefix_space,
                @JsonProperty("trim_offsets") Boolean trim_offsets,
                @JsonProperty("use_regex") Boolean use_regex) {
            this.type = type;
            this.pattern = pattern;
            this.behavior = behavior;
            this.invert = invert;
            this.individual_digits = individual_digits;
            this.add_prefix_space = add_prefix_space;
            this.trim_offsets = trim_offsets;
            this.use_regex = use_regex;
        }

        public List<String> pretokenize(String sentence) {
            switch (type) {
                case "Split":
                    return splitRegex(sentence);
                case "Digits":
                    return splitDigits(sentence);
                case "ByteLevel":
                    return Collections.singletonList(sentence);
                default:
                    throw new IllegalArgumentException("Invalid pre-tokenizer type: " + type);
            }
        }

        private List<String> splitRegex(String s) {
            Matcher m = pattern.regex.matcher(s);
            List<String> ret = new ArrayList<>();
            int start = 0;
            while (m.find()) {
                ret.add(s.substring(start, m.start()));
                ret.add(m.group());
                start = m.end();
            }

            String p = start >= s.length() ? "" : s.substring(start);
            if (!p.isEmpty()) ret.add(p);
            return ret;
        }

        private List<String> splitDigits(String sentence) {
            return List.of(sentence.split("(?<=\\D)(?=\\d)|(?<=\\d)(?=\\D)"));
        }
    }

    // Pattern class
    public static class Pattern {
        public final java.util.regex.Pattern regex;

        @JsonCreator
        public Pattern(@JsonProperty("Regex") String regex) {
            this.regex = java.util.regex.Pattern.compile(regex);
        }
    }
}
