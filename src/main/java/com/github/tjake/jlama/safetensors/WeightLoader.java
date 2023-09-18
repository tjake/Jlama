package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.tensor.AbstractTensor;

public interface WeightLoader {
    AbstractTensor load(String name);

    DType getModelDType();
}
