package com.github.tjake.jlama.safetensors.tokenizer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import com.google.common.base.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.tjake.jlama.safetensors.SafeTensorSupport;

public abstract class BPETokenizer implements Tokenizer {
    protected static final Logger logger = LoggerFactory.getLogger(BPETokenizer.class);
    protected final TokenizerModel model;
    protected final ByteBuffer decodeBuffer = ByteBuffer.allocate(4);

    protected BPETokenizer(Path modelRoot) {
        Preconditions.checkArgument(modelRoot.resolve("tokenizer.json").toFile().exists(), "No tokenizer.jsom found in " + modelRoot);

        try {
            this.model = SafeTensorSupport.loadTokenizer(modelRoot);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<String> tokenize(String sentence) {

        if (sentence.isEmpty())
            return Collections.emptyList();

        if (model.preTokenizer() != null)
            return model.preTokenizer().pretokenize(sentence);

        return Collections.singletonList(sentence);
    }

    protected String preProcess(String sentence) {
        return sentence;
    }

    @Override
    public long[] encode(String rawSentence) {

        List<String> sentencePieces = tokenize(rawSentence);
        List<Long> allTokens = new ArrayList<>();

        for (String sentence : sentencePieces) {
            List<Long> tokens = new ArrayList<>();
            sentence = preProcess(sentence);
            int[] codes = sentence.codePoints().toArray();
            for (int i = 0; i < codes.length; i++) {
                String c = Character.toString(codes[i]);
                Long id = model.vocabLookup.get(c);
                if (id != null) {
                    // we found this codepoint in vocab, add it as a token
                    //logger.debug("{} -> {}", c, id);
                    tokens.add(id);
                } else {
                    if (model.byteFallback) {
                        // byte_fallback encoding: just encode each byte as a token
                        String code = Character.toString(codes[i]);
                        byte[] chars = code.getBytes(StandardCharsets.UTF_8);
                        for (int k = 0; k < chars.length; k++) {
                            long token = encodeCharacterAsToken(chars[k]);
                            //logger.debug("byte {} -> {}", Byte.toUnsignedInt(chars[k]), token);
                            tokens.add(token);
                        }
                    } else {
                        if (model.unkToken != null) {
                            tokens.add(model.vocabLookup.get(model.unkToken));
                        }
                    }
                }
            }

            // merge the best consecutive pair each iteration
            while (true) {
                long bestId = -1;
                long bestIdx = -1;

                for (int i = 0; i < tokens.size() - 1; i++) {
                    // check if we can merge the pair (tokens[i], tokens[i+1])
                    String merge = String.format("%s%s", decodeInternal(tokens.get(i)), decodeInternal(tokens.get(i + 1)));
                    Long id = model.vocabLookup.get(merge);
                    if (id != null) {
                        // this merge pair exists in vocab! record its position
                        bestId = id;
                        bestIdx = i;
                        break;
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
        if (decoded == null)
            decoded = model.unkToken;

        return decoded;
    }

    @Override
    public String decode(long id) {
        return maybeDecodeTokenAsCharacter(id).map(c -> {
            // We have a continuation byte or are buffering them
            if (Character.isUnicodeIdentifierPart(c) || decodeBuffer.remaining() < 4) {
                decodeBuffer.put((byte)c.charValue());

                //Unicode symbol is ready
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

    //Only used for merging
    protected String decodeInternal(long id) {
        return maybeDecodeTokenAsCharacter(id).map(Object::toString).orElseGet(() -> {

            String s = model.vocabLookup.inverse().get(id);
            if (s == null)
                s = model.unkToken;

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
}
