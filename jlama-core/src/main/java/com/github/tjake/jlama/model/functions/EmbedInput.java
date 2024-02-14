package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.tensor.AbstractTensor;

public interface EmbedInput {
    AbstractTensor inputTokenToEmbedding(int inputToken, int position);
}
