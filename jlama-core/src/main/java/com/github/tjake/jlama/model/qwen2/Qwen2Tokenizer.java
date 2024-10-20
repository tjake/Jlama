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
