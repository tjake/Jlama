package com.github.tjake.llmj.safetensors;

import com.github.tjake.llmj.model.Tensor;

public interface WeightLoader {
    Tensor load(String name);
}
