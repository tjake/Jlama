package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.Optional;

public interface ClassifyOutput {
    public AbstractTensor getClassificationWeights();
    public Optional<AbstractTensor> getClassificationBias();
}
