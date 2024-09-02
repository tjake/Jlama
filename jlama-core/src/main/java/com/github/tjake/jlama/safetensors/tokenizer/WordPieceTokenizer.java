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

import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.google.common.base.Preconditions;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * WordPiece tokenizer
 * @see <a href="https://github.com/google-research/bert/blob/master/tokenization.py">...</a>
 */
public class WordPieceTokenizer implements Tokenizer {

    protected final TokenizerModel model;
    protected final PromptSupport promptSupport;
    protected final long sepToken;
    protected final long clsToken;
    protected final long unkToken;

    protected static final String sepString = "[SEP]";
    protected static final String clsString = "[CLS]";
    protected static final String unkString = "[UNK]";

    public WordPieceTokenizer(Path modelRoot) {
        Preconditions.checkArgument(modelRoot.resolve("tokenizer.json").toFile().exists(), "No tokenizer.json found in " + modelRoot);

        try {
            this.model = SafeTensorSupport.loadTokenizer(modelRoot);
            Preconditions.checkArgument(
                model.type == null || model.type.equalsIgnoreCase("WordPiece"),
                "Invalid model type: " + model.type
            );

            this.promptSupport = new PromptSupport(model);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        this.sepToken = model.vocabLookup.get(sepString);
        this.clsToken = model.vocabLookup.get(clsString);
        this.unkToken = model.vocabLookup.get(unkString);
    }

    @Override
    public TokenizerModel getModel() {
        return model;
    }

    @Override
    public List<String> tokenize(String sentence) {

        sentence = preProcess(sentence);

        String[] whitespaceSplits = sentence.split("\\s+");

        List<String> tokens = new ArrayList<>();
        tokens.add(clsString);

        List<String> stringList = Arrays.stream(whitespaceSplits)
            .flatMap(this::splitByPunctuation)
            .map(str -> str.length() > 200 ? model.unkToken : str)
            .flatMap(str -> {
                boolean isBad = false;
                List<String> subTokens = new ArrayList<>();

                int start = 0;
                while (start < str.length()) {
                    int end = str.length();
                    String curSubStr = null;
                    while (start < end) {
                        String substr = str.substring(start, end);
                        if (start > 0) substr = "##" + substr;
                        if (model.vocabLookup.containsKey(substr)) {
                            curSubStr = substr;
                            break;
                        }
                        end -= 1;
                    }
                    if (curSubStr == null) {
                        isBad = true;
                        break;
                    }

                    subTokens.add(curSubStr);
                    start = end;
                }

                if (isBad) subTokens.add(model.unkToken);

                return subTokens.stream();
            })
            .collect(Collectors.toList());

        tokens.addAll(stringList);
        tokens.add(sepString);

        return tokens;
    }

    protected String preProcess(String sentence) {
        sentence = sentence.toLowerCase().strip();

        return cleanText(sentence);
    }

    static boolean isControl(Integer c) {
        // These are technically control characters but we count them as whitespace characters.
        if (c == '\t' || c == '\n' || c == '\r') return false;

        return Character.isISOControl(c);
    }

    static boolean isPunctuation(Integer cp) {
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
            return true;
        }

        int t = Character.getType(cp);
        if (t >= 20 && t <= 24) return true;

        return false;
    }

    String cleanText(String sentence) {
        return sentence.codePoints().map(c -> {
            if (c == 0 || c == 0xfffd || isControl(c)) return -1;

            if (Character.isWhitespace(c)) return ' ';

            return c;
        }).filter(c -> c != -1).mapToObj(Character::toString).collect(Collectors.joining());
    }

    Stream<String> splitByPunctuation(String str) {
        List<String> result = new ArrayList<>();

        int start = 0;

        for (int offset = 0; offset < str.length();) {
            int codepoint = str.codePointAt(offset);

            if (isPunctuation(codepoint)) {
                if (offset != start) {
                    result.add(str.substring(start, offset));
                }
                result.add(str.substring(offset, offset + Character.charCount(codepoint)));
                start = offset + Character.charCount(codepoint);
            }

            offset += Character.charCount(codepoint);
        }

        // Add the remaining part if there's any
        if (start != str.length()) {
            result.add(str.substring(start));
        }

        return result.stream();
    }

    @Override
    public long[] encode(String sentence) {
        return tokenize(sentence).stream().mapToLong(s -> model.vocabLookup.get(s)).toArray();
    }

    protected String postProcessToken(String decoded) {
        if (decoded.startsWith("##")) return decoded.substring(2);

        return " " + decoded;
    }

    @Override
    public String decode(long id) {
        return postProcessToken(model.vocabLookup.inverse().get(id));
    }

    protected String postProcess(String sentence) {
        return sentence.strip();
    }

    @Override
    public String decode(long[] ids) {
        return postProcess(Arrays.stream(ids).mapToObj(this::decode).collect(Collectors.joining()));
    }

    @Override
    public Optional<PromptSupport> promptSupport() {
        return model.promptTemplates().isPresent() ? Optional.of(promptSupport) : Optional.empty();
    }
}
