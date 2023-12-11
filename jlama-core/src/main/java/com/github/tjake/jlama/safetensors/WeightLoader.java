package com.github.tjake.jlama.safetensors;

import java.util.Map;
import java.util.Optional;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;

public interface WeightLoader extends AutoCloseable {

    Map<String, String> metadata();

    Map<String, TensorInfo> tensorInfoMap();

    default AbstractTensor load(String name) {
        return load(name, Optional.empty());
    }

    AbstractTensor load(String name, Optional<Pair<Integer, Integer>> offset);

    DType getModelDType();
}
