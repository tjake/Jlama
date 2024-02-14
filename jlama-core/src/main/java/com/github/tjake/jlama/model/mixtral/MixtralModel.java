package com.github.tjake.jlama.model.mixtral;


import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.mistral.MistralModel;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

public class MixtralModel extends MistralModel {
    private static final Logger logger = LoggerFactory.getLogger(MixtralModel.class);
    
    public MixtralModel(Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public MixtralModel(InferenceType inferenceType, Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {

        MixtralConfig mixtralConfig = (MixtralConfig) c;
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            logger.info("Quantizing model with {} - Please hold...", qType);
        }

        TransformerBlock[] transformerBlocks = new TransformerBlock[c.getNumberOfLayers()];

        IntStream.range(c.layerStart(), c.layerEnd()).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(this,
                    weights.load(prefix + "q_proj.weight", c.offset()).quantize(qType),
                    weights.load(prefix + "k_proj.weight", c.offset()).quantize(qType),
                    weights.load(prefix + "v_proj.weight", c.offset()).quantize(qType),
                    weights.load(prefix + "o_proj.weight", c.offset()).quantize(qType));

            prefix = base + "block_sparse_moe.";

            AbstractTensor[] expertGateWeights = new AbstractTensor[mixtralConfig.numberOfExperts];
            AbstractTensor[] expertDownWeights = new AbstractTensor[mixtralConfig.numberOfExperts];
            AbstractTensor[] expertUpWeights = new AbstractTensor[mixtralConfig.numberOfExperts];

            for (int e = 0; e < mixtralConfig.numberOfExperts; e++) {
                String expertPrefix = prefix + "experts." + e + ".";
                expertGateWeights[e] = weights.load(expertPrefix + "w1.weight", c.offset()).quantize(qType);
                expertDownWeights[e] = weights.load(expertPrefix + "w2.weight").quantize(qType);
                expertUpWeights[e] = weights.load(expertPrefix + "w3.weight", c.offset()).quantize(qType);
            }


            MOEBlock moe = new MOEBlock(this, mixtralConfig.numberOfExperts, mixtralConfig.numberOfExpertsPerToken, ActivationFunction.Type.SILU,
                    weights.load(prefix + "gate.weight", c.offset()).quantize(qType),
                    expertGateWeights, //w1
                    expertDownWeights, //w2
                    expertUpWeights);  //w3

            transformerBlocks[i] = new TransformerBlock( this,
                    new RMSNorm(this, weights.load(base + "input_layernorm.weight", c.offset()).quantize(qType)),
                    attention,
                    new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight", c.offset()).quantize(qType)),
                    moe);
        });

        return transformerBlocks;
    }
}
