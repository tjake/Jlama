package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Used to define a feed forward function like MLP or MOE
 */
public interface FeedForward {

    AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer);
}
