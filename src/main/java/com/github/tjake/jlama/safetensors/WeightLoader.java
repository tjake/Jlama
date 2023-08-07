package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.model.Tensor;

public interface WeightLoader {
    Tensor load(String name);
}
