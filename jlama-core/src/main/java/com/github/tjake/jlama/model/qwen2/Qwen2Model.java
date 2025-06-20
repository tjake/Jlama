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
package com.github.tjake.jlama.model.qwen2;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

public class Qwen2Model extends LlamaModel {

    private static final Logger logger = LoggerFactory.getLogger(Qwen2Model.class);

    public Qwen2Model(
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public Qwen2Model(
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
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            logger.info("Quantizing model with {} - Please hold...", qType);
        }

        TransformerBlock[] transformerBlocks = new TransformerBlock[c.dctx().numberOfLayers];

        IntStream.range(c.dctx().layerStart, c.dctx().layerEnd).parallel().forEach(i -> {

            int relativeLayer = i - c.dctx().layerStart; // FIXME: add a helper to the context

            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(
                this,
                relativeLayer,
                Optional.of(weights.load(prefix + "q_proj.bias").quantize(qType)),
                Optional.of(weights.load(prefix + "k_proj.bias").quantize(qType)),
                Optional.of(weights.load(prefix + "v_proj.bias").quantize(qType)),
                weights.load(prefix + "q_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "k_proj.weight", c.dctx(), true, false).quantize(qType),
                weights.load(prefix + "v_proj.weight", c.dctx(), true, false).quantize(qType),
                Optional.empty(),
                weights.load(prefix + "o_proj.weight", c.dctx(), false, true).quantize(qType)
            );

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(
                this,
                c.activationFunction,
                weights.load(prefix + "gate_proj.weight", c.dctx(), true, false).quantize(qType), // w1
                weights.load(prefix + "down_proj.weight", c.dctx(), false, true).quantize(qType), // w2
                weights.load(prefix + "up_proj.weight", c.dctx(), true, false).quantize(qType)
            ); // w3

            transformerBlocks[relativeLayer] = new TransformerBlock(
                this,
                relativeLayer,
                new RMSNorm(this, weights.load(base + "input_layernorm.weight").quantize(qType)),
                attention,
                new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight").quantize(qType)),
                mlp
            );
        });

        return transformerBlocks;
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("QWEN2");
    }
}
