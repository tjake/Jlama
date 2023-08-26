package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.model.AbstractTensor;

public interface WeightLoader {
    AbstractTensor load(String name);

    DType getModelDType();
}
