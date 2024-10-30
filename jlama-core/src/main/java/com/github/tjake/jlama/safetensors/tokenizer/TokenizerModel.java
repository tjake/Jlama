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

import static com.github.tjake.jlama.safetensors.tokenizer.BPETokenizer.alteredBytes;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.google.common.base.Preconditions;
import com.google.common.collect.*;

import java.text.Normalizer;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tokenizer model, loosely based on Huggingface's Tokenizer format
 *
 * @see <a href="https://huggingface.co/transformers/main_classes/tokenizer.html">Huggingface Tokenizer</a>
 *
 * This class also holds the prompt templates
 * @see <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models">Chat Templating</a>
 * @see PromptSupport
 */
public class TokenizerModel {
    private static final Logger logger = LoggerFactory.getLogger(TokenizerModel.class);
    private static final java.util.regex.Pattern gpt2Pattern = java.util.regex.Pattern.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    );

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

    @JsonProperty("merges")
    public final Map<String, Long> merges;

    private PreTokenizer preTokenizer;
    private Normalizer normalizer;
    private BiMap<String, Long> addedTokens = HashBiMap.create();
    private BiMap<String, Long> specialTokens = HashBiMap.create();

    private java.util.regex.Pattern addedTokenPattern;

    // This is pretty much a hack to support the legacy tokenizer
    private boolean legacy = false;

    private Optional<Map<String, String>> promptTemplates = Optional.empty();
    private boolean hasToolSupport = false;
    private String eosToken = "";
    private String bosToken = "";
    private final boolean ignoreMerges;

    @JsonCreator
    public TokenizerModel(
        @JsonProperty("type") String type,
        @JsonProperty("unk_token") String unkToken,
        @JsonProperty("fuse_unk") boolean fuseUnk,
        @JsonProperty("byte_fallback") boolean byteFallback,
        @JsonProperty("vocab") Map<String, Long> vocabLookup,
        @JsonProperty("ignore_merges") Boolean ignoreMerges,
        @JsonProperty("merges") List<Object> merges
    ) {
        this.type = type;
        this.unkToken = unkToken;
        this.fuseUnk = fuseUnk;
        this.byteFallback = byteFallback;
        this.vocabLookup = HashBiMap.create(vocabLookup);
        this.ignoreMerges = ignoreMerges != null && ignoreMerges;
        this.merges = new HashMap<>();
        if (merges != null) {
            for (int i = 0; i < merges.size(); i++) {
                if (merges.get(i) instanceof String) {
                    this.merges.put((String) merges.get(i), (long) i);
                } else if (merges.get(i) instanceof List) {
                    List<String> merge = (List<String>) merges.get(i);
                    this.merges.put(merge.get(0) + " " + merge.get(1), (long) i);
                } else {
                    throw new IllegalArgumentException("Invalid merge format: " + merges.get(i));
                }
            }
        }
    }

    public PreTokenizer preTokenizer() {
        return preTokenizer;
    }

    public void setPreTokenizer(PreTokenizer preTokenizer) {
        if (preTokenizer != null) {
            this.preTokenizer = preTokenizer;
            this.legacy = preTokenizer.isLegacy;
        }
    }

    public Normalizer normalizer() {
        return normalizer;
    }

    public void setNormalizer(Normalizer normalizer) {
        this.normalizer = normalizer;
    }

    public void setAddedTokens(List<Map<String, Object>> addedTokens) {
        if (addedTokens != null && !addedTokens.isEmpty()) {
            for (Map<String, Object> token : addedTokens) {
                this.addedTokens.put((String) token.get("content"), ((Integer) token.get("id")).longValue());
                this.vocabLookup.put((String) token.get("content"), ((Integer) token.get("id")).longValue());
                if (token.containsKey("special") && (Boolean) token.get("special")) {
                    this.specialTokens.put((String) token.get("content"), ((Integer) token.get("id")).longValue());
                }
            }

            // Lock down the added tokens
            this.addedTokens = ImmutableBiMap.copyOf(this.addedTokens);
            this.specialTokens = ImmutableBiMap.copyOf(this.specialTokens);

            // Create a regular expression from the list of delimiters
            StringBuilder regex = new StringBuilder();
            List<String> delimiters = new ArrayList<>(this.addedTokens.keySet());

            for (int i = 0; i < delimiters.size(); i++) {
                if (i != 0) {
                    regex.append("|");
                }
                regex.append(java.util.regex.Pattern.quote(delimiters.get(i)));
            }

            this.addedTokenPattern = java.util.regex.Pattern.compile(regex.toString());
        }
    }

    public boolean ignoreMerges() {
        return ignoreMerges;
    }

    public Map<String, Long> addedTokens() {
        return addedTokens;
    }

    public java.util.regex.Pattern addedTokenPattern() {
        return addedTokenPattern;
    }

    public boolean isLegacy() {
        return legacy;
    }

    public void setLegacy(boolean legacy) {
        this.legacy = legacy;
    }

    public Optional<Map<String, String>> promptTemplates() {
        return promptTemplates;
    }

    public void setPromptTemplates(Map<String, String> promptTemplates) {

        if (promptTemplates != null) {

            hasToolSupport = promptTemplates.values().stream().anyMatch(s -> s.toLowerCase().contains("tools"));

            this.promptTemplates = Optional.of(promptTemplates);
        }
    }

    public boolean hasToolSupport() {
        return hasToolSupport;
    }

    public void setEosToken(String eosToken) {
        this.eosToken = eosToken;
    }

    public String eosToken() {
        return eosToken;
    }

    public void setBosToken(String bosToken) {
        this.bosToken = bosToken;
    }

    public String bosToken() {
        return bosToken;
    }

    public boolean isSpecialToken(long token) {
        return specialTokens.containsValue(token);
    }

    public boolean isSpecialToken(String token) {
        return specialTokens.containsKey(token);
    }

    // Splitter for added token pattern (optionally with delimiters)
    static String[] split(java.util.regex.Pattern p, CharSequence input, int limit, boolean withDelimiters) {
        int matchCount = 0;
        int index = 0;
        boolean matchLimited = limit > 0;
        ArrayList<String> matchList = new ArrayList<>();
        Matcher m = p.matcher(input);

        // Add segments before each match found
        while (m.find()) {
            if (!matchLimited || matchCount < limit - 1) {
                if (index == 0 && index == m.start() && m.start() == m.end()) {
                    // no empty leading substring included for zero-width match
                    // at the beginning of the input char sequence.
                    continue;
                }
                String match = input.subSequence(index, m.start()).toString();
                matchList.add(match);
                index = m.end();
                if (withDelimiters) {
                    matchList.add(input.subSequence(m.start(), index).toString());
                }
                ++matchCount;
            } else if (matchCount == limit - 1) { // last one
                String match = input.subSequence(index, input.length()).toString();
                matchList.add(match);
                index = m.end();
                ++matchCount;
            }
        }

        // If no match was found, return this
        if (index == 0) return new String[] { input.toString() };

        // Add remaining segment
        if (!matchLimited || matchCount < limit) matchList.add(input.subSequence(index, input.length()).toString());

        // Construct result
        int resultSize = matchList.size();
        if (limit == 0) {
            while (resultSize > 0 && matchList.get(resultSize - 1).isEmpty()) {
                resultSize--;
            }
        }
        String[] result = new String[resultSize];
        return matchList.subList(0, resultSize).toArray(result);
    }

    public static class Normalizer {
        public final String type;

        public final List<NormalizerItem> normalizerItems;

        @JsonCreator
        public Normalizer(@JsonProperty("type") String type, @JsonProperty("normalizers") List<NormalizerItem> normalizerItems) {
            this.type = type;
            this.normalizerItems = normalizerItems == null ? Collections.emptyList() : ImmutableList.copyOf(normalizerItems);
        }

        public String normalize(String sentence) {
            if (normalizerItems.isEmpty()) return sentence;

            Preconditions.checkArgument(type.equalsIgnoreCase("Sequence"), "Invalid normalizer type: " + type);
            for (NormalizerItem item : normalizerItems) {
                sentence = item.normalize(sentence);
            }

            return sentence;
        }
    }

    public static class NormalizerItem {
        public final String type;
        public final String prepend;
        public final Map<String, String> pattern;
        public final String content;

        @JsonCreator
        public NormalizerItem(
            @JsonProperty("type") String type,
            @JsonProperty("prepend") String prepend,
            @JsonProperty("pattern") Map<String, String> pattern,
            @JsonProperty("content") String content
        ) {
            this.type = type;
            this.prepend = prepend;
            this.pattern = pattern;
            this.content = content;
        }

        public String normalize(String sentence) {
            switch (type) {
                case "Replace":
                    return replace(sentence);
                case "Prepend":
                    return prepend(sentence);
                case "NFC":
                case "NFKC":
                case "NFD":
                case "NFKD":
                    return formNormalize(sentence);
                default:
                    throw new IllegalArgumentException("Invalid normalizer type: " + type);
            }
        }

        private String formNormalize(String sentence) {
            java.text.Normalizer.Form form = java.text.Normalizer.Form.valueOf(type);
            return java.text.Normalizer.normalize(sentence, form);
        }

        private String replace(String sentence) {
            for (Map.Entry<String, String> entry : pattern.entrySet()) {
                if (!entry.getKey().equalsIgnoreCase("String")) logger.warn("Ignoring unknown pattern key: " + entry.getKey());
                sentence = sentence.replaceAll(entry.getValue(), content);
            }

            return sentence;
        }

        private String prepend(String sentence) {
            return prepend + sentence;
        }
    }

    // PreTokenizer class
    public static class PreTokenizer {
        public final String type;
        public final String replacement;
        public final String prependScheme;
        public final boolean isLegacy;
        public final List<PretokenizerItem> pretokenizers;

        @JsonCreator
        public PreTokenizer(
            @JsonProperty("type") String type,
            @JsonProperty("replacement") String replacement,
            @JsonProperty("prepend_scheme") String prependScheme,
            @JsonProperty("pretokenizers") List<PretokenizerItem> pretokenizers
        ) {
            this.type = type;
            this.replacement = replacement;
            this.prependScheme = prependScheme;
            this.pretokenizers = pretokenizers == null ? Collections.emptyList() : ImmutableList.copyOf(pretokenizers);
            this.isLegacy = this.pretokenizers.stream().map(p -> p.type).anyMatch(t -> t.equals("ByteLevel"));
        }

        public List<String> pretokenize(String sentence) {

            if (type.equalsIgnoreCase("MetaSpace")) {
                if (prependScheme.equalsIgnoreCase("first")) {
                    sentence = " " + sentence;
                }

                return Collections.singletonList(sentence.replaceAll("[ \t]+", replacement));
            }

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
            @JsonProperty("use_regex") Boolean use_regex
        ) {
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
                    // if (use_regex) return splitGpt2(sentence);
                    // Rather than deal with this, we'll just force byte fallback (only difference is how unk is
                    // handled)
                    return Collections.singletonList(sentence);
                default:
                    throw new IllegalArgumentException("Invalid pre-tokenizer type: " + type);
            }
        }

        private List<String> byteLevel(String sentence) {
            return List.of(
                sentence.codePoints().map(c -> alteredBytes.getOrDefault(c, c)).mapToObj(Character::toString).collect(Collectors.joining())
            );
        }

        private List<String> splitGpt2(String sentence) {
            return List.of(gpt2Pattern.split(sentence));
        }

        private List<String> splitRegex(String s) {
            Matcher m = pattern.regex.matcher(s);
            List<String> ret = new ArrayList<>();
            int start = 0;
            while (m.find()) {
                String r = s.substring(start, m.start());
                if (!r.isEmpty()) ret.add(r);

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
