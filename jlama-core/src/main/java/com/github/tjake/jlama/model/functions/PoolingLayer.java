package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.Optional;

public interface PoolingLayer {
    AbstractTensor getPoolingWeights();
    Optional<AbstractTensor> getPoolingBias();
}
