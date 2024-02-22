package com.github.tjake.jlama.model.gemma;

import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Collectors;

import com.github.tjake.jlama.safetensors.tokenizer.BPETokenizer;

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
    protected Optional<Character> maybeDecodeTokenAsCharacter(long id)
    {
        //Handle ascii codes (shifted in vocab)
        if (model.byteFallback && id >= byteFallbackEncodingOffset && id < 256 + byteFallbackEncodingOffset) {
            char c = (char)(id - byteFallbackEncodingOffset);
            return Optional.of(c);
        }

        return Optional.empty();
    }

    @Override
    protected String preProcess(String sentence)
    {
        return sentence.replace(" ", SPIECE_UNDERLINE);
    }

    @Override
    protected String postProcess(String sentence) {
        return sentence.stripLeading();
    }

    @Override
    protected String postProcessToken(String decoded) {
        if (decoded == null)
            decoded = model.unkToken;

        decoded = decoded.replaceAll("</?s>", "");
        decoded = decoded.replaceAll(SPIECE_UNDERLINE, " ");

        return decoded;
    }
}
