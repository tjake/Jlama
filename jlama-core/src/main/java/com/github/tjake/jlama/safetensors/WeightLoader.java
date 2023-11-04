package com.github.tjake.jlama.safetensors;

import java.util.Map;

import com.github.tjake.jlama.tensor.AbstractTensor;

public interface WeightLoader extends AutoCloseable {

    Map<String, String> metadata();

    Map<String, TensorInfo> tensorInfoMap();

    AbstractTensor load(String name);

    DType getModelDType();
}
