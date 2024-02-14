package com.github.tjake.jlama.model.functions;

import java.util.Optional;
import java.util.UUID;
import java.util.function.BiConsumer;

public interface Generator {
    default void generate(UUID session, String prompt, float temperature, int ntokens, boolean useEOS, BiConsumer<String, Float> onTokenWithTimings) {
        generate(session, prompt, null, temperature, ntokens, useEOS, onTokenWithTimings);
    }

    void generate(UUID session, String prompt, String cleanPrompt, float temperature, int ntokens, boolean useEOS, BiConsumer<String, Float> onTokenWithTimings);

    String wrapPrompt(String prompt, Optional<String> systemPrompt);
}