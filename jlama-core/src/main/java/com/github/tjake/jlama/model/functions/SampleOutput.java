package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.model.LayerNorm;
import com.github.tjake.jlama.tensor.AbstractTensor;

public interface SampleOutput {

    LayerNorm getOutputLayerNorm();

    AbstractTensor getOutputLogitsWeights();

}
