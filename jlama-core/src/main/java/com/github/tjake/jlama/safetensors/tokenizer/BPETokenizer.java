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
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableBiMap;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Byte Pair Encoding tokenizer
 */
public abstract class BPETokenizer implements Tokenizer {
    protected static final Logger logger = LoggerFactory.getLogger(BPETokenizer.class);
    protected final TokenizerModel model;
    protected final PromptSupport promptSupport;
    protected final ByteBuffer decodeBuffer = ByteBuffer.allocate(4);

    public static BiMap<Integer, Integer> alteredBytes; // Codepoint and Token mapping needed for legacy mode

    static {
        // https://github.com/openai/gpt-2/blob/master/src/encoder.py#L19
        BiMap<Integer, Integer> tmpAlteredBytes = HashBiMap.create();
        int i = 0;
        for (int c = 0; c < 256; c++) {
            if ((c < '!' || c > '~') && (c < '¡' || c > '¬') && (c < '®' || c > 'ÿ')) {
                int codepoint = (i++ + 256);
                tmpAlteredBytes.put(c, codepoint);
            }
        }

        alteredBytes = ImmutableBiMap.copyOf(tmpAlteredBytes);
    }

    protected BPETokenizer(Path modelRoot) {
        Preconditions.checkArgument(modelRoot.resolve("tokenizer.json").toFile().exists(), "No tokenizer.json found in " + modelRoot);

        try {
            this.model = SafeTensorSupport.loadTokenizer(modelRoot);
            this.promptSupport = new PromptSupport(model);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public TokenizerModel getModel() {
        return model;
    }

    @Override
    public List<String> tokenize(String sentence) {

        if (sentence.isEmpty()) return Collections.emptyList();

        if (model.preTokenizer() == null && model.addedTokenPattern() == null) Collections.singletonList(sentence);

        List<String> sentencePieces = new ArrayList<>();
        if (model.addedTokenPattern() != null) {
            // Split the sentence into pieces using the added token pattern
            // Any non-added token is split into pieces using the pre-tokenizer
            String[] pieces = TokenizerModel.split(model.addedTokenPattern(), sentence, 0, true);
            for (String piece : pieces) {
                if (!piece.isEmpty()) {
                    if (model.addedTokens().containsKey(piece)) sentencePieces.add(piece);
                    else if (model.preTokenizer() != null) sentencePieces.addAll(model.preTokenizer().pretokenize(piece));
                    else sentencePieces.add(piece);
                }
            }
        } else if (model.preTokenizer() != null) {
            sentencePieces.addAll(model.preTokenizer().pretokenize(sentence));
        } else {
            sentencePieces.add(sentence);
        }

        return sentencePieces;
    }

    protected String preProcess(String sentence) {
        return sentence;
    }

    @Override
    public long[] encode(String rawSentence) {

        List<String> sentencePieces = tokenize(rawSentence);
        List<Long> allTokens = new ArrayList<>();

        for (String sentence : sentencePieces) {
            if (model.addedTokens() != null && model.addedTokens().containsKey(sentence)) {
                allTokens.add(model.addedTokens().get(sentence));
                continue;
            }
            List<Long> tokens = new ArrayList<>();
            sentence = preProcess(sentence);
            int[] codes = sentence.codePoints().toArray();
            for (int i = 0; i < codes.length; i++) {
                String c = Character.toString(codes[i]);
                Long id = model.vocabLookup.get(c);
                if (id != null) {
                    // we found this codepoint in vocab, add it as a token
                    // logger.debug("{} -> {}", c, id);
                    tokens.add(id);
                } else {
                    if (model.byteFallback) {
                        // byte_fallback encoding: just encode each byte as a token
                        String code = Character.toString(codes[i]);
                        byte[] chars = code.getBytes(StandardCharsets.UTF_8);
                        for (int k = 0; k < chars.length; k++) {
                            long token = encodeCharacterAsToken(chars[k]);
                            // logger.debug("byte {} -> {}", Byte.toUnsignedInt(chars[k]), token);
                            tokens.add(token);
                        }
                    } else {
                        if (model.unkToken != null) {
                            tokens.add(model.vocabLookup.get(model.unkToken));
                        }
                    }
                }
            }

            // merge the best consecutive tuple each iteration,
            // until we can't find any more pairs to merge
            while (true) {
                long bestId = -1;
                long bestIdx = -1;
                long bestRank = Long.MAX_VALUE;

                for (int i = 0; i < tokens.size() - 1; i++) {
                    // check if we can merge the pair (tokens[i], tokens[i+1])
                    String token1 = decodeInternal(tokens.get(i));
                    String token2 = decodeInternal(tokens.get(i + 1));

                    String merge2 = String.format("%s %s", token1, token2);
                    String merge3 = String.format("%s%s", token1, token2);

                    if (model.merges.containsKey(merge2)) {
                        Long id = model.vocabLookup.get(merge3);
                        if (id != null) {
                            // Check if this merge has a better rank (i.e., lower rank number)
                            long rank = model.merges.get(merge2);
                            if (rank < bestRank) {
                                // this merge pair exists in vocab! record its position
                                bestId = id;
                                bestIdx = i;
                                bestRank = rank;
                            }
                        }
                    }
                }

                if (bestIdx == -1) {
                    break; // we couldn't find any more pairs to merge, so we're done
                }

                // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
                tokens.set((int) bestIdx, bestId);
                // delete token at position best_idx+1, shift the entire sequence back 1
                tokens.remove((int) bestIdx + 1);
            }

            allTokens.addAll(tokens);
        }

        return allTokens.stream().mapToLong(s -> s).toArray();
    }

    protected String postProcessToken(String decoded) {
        if (decoded == null) decoded = model.unkToken;

        return decoded;
    }

    @Override
    public String decode(long id) {
        return maybeDecodeTokenAsCharacter(id).map(c -> {
            // We have a continuation byte or are buffering them
            if (Character.isUnicodeIdentifierPart(c) || decodeBuffer.remaining() < 4) {
                decodeBuffer.put((byte) c.charValue());

                // Unicode symbol is ready
                if (decodeBuffer.remaining() == 0) {
                    String s = new String(decodeBuffer.array());
                    decodeBuffer.rewind();
                    return s;
                }

                return "";
            }
            return Character.toString(c);
        }).orElseGet(() -> postProcessToken(model.vocabLookup.inverse().get(id)));
    }

    protected abstract long encodeCharacterAsToken(byte c);

    protected abstract Optional<Character> maybeDecodeTokenAsCharacter(long id);

    // Only used for merging
    protected String decodeInternal(long id) {
        return maybeDecodeTokenAsCharacter(id).map(Object::toString).orElseGet(() -> {
            String s = model.vocabLookup.inverse().get(id);
            if (s == null) s = model.unkToken;

            return s;
        });
    }

    protected String postProcess(String sentence) {
        return sentence;
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
