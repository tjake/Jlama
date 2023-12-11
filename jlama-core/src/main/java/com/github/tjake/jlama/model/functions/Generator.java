package com.github.tjake.jlama.model.functions;

import java.util.function.BiConsumer;

public interface Generator {
    default void generate(String prompt, float temperature, int ntokens, boolean useEOS, BiConsumer<String, Float> onTokenWithTimings) {
        generate(prompt, null, temperature, ntokens, useEOS, onTokenWithTimings);
    }

    void generate(String prompt, String cleanPrompt, float temperature, int ntokens, boolean useEOS, BiConsumer<String, Float> onTokenWithTimings);
}
