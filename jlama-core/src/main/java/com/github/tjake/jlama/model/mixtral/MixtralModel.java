/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.model.mixtral;

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

    public MixtralModel(
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public MixtralModel(
        InferenceType inferenceType,
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("MIXTRAL");
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {

        MixtralConfig mixtralConfig = (MixtralConfig) c;
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            logger.info("Quantizing model with {} - Please hold...", qType);
        }

        TransformerBlock[] transformerBlocks = new TransformerBlock[c.dctx().numberOfLayers];

        IntStream.range(c.dctx().layerStart, c.dctx().layerEnd).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(
                this,
                i,
                weights.load(prefix + "q_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "k_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "v_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "o_proj.weight").quantize(qType)
            );

            prefix = base + "block_sparse_moe.";

            AbstractTensor[] expertGateWeights = new AbstractTensor[mixtralConfig.numberOfExperts];
            AbstractTensor[] expertDownWeights = new AbstractTensor[mixtralConfig.numberOfExperts];
            AbstractTensor[] expertUpWeights = new AbstractTensor[mixtralConfig.numberOfExperts];

            for (int e = 0; e < mixtralConfig.numberOfExperts; e++) {
                String expertPrefix = prefix + "experts." + e + ".";
                expertGateWeights[e] = weights.load(expertPrefix + "w1.weight", c.dctx(), true, false).quantize(qType);
                expertDownWeights[e] = weights.load(expertPrefix + "w2.weight").quantize(qType);
                expertUpWeights[e] = weights.load(expertPrefix + "w3.weight", c.dctx(), true, false).quantize(qType);
            }

            MoEBlock moe = new MoEBlock(
                this,
                mixtralConfig.numberOfExperts,
                mixtralConfig.numberOfExpertsPerToken,
                c.activationFunction,
                weights.load(prefix + "gate.weight").quantize(qType),
                expertGateWeights, // w1
                expertDownWeights, // w2
                expertUpWeights
            ); // w3

            transformerBlocks[i] = new TransformerBlock(
                this,
                i,
                new RMSNorm(this, weights.load(base + "input_layernorm.weight").quantize(qType)),
                attention,
                new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight").quantize(qType)),
                moe
            );
        });

        return transformerBlocks;
    }
}
