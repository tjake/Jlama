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
package com.github.tjake.jlama.model.gpt2;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.Optional;

public class GPT2Model extends AbstractModel {

    public GPT2Model(Config c, WeightLoader w, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(InferenceType.FULL_GENERATION, c, w, tokenizer, workingDType, workingQType, modelQType);
    }

    public GPT2Model(
        InferenceType inferenceType,
        Config c,
        WeightLoader w,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType);
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("GPT2");
    }

    @Override
    protected EmbedInput loadInputWeights() {
        final AbstractTensor wte = weights.load("wte.weight");
        final AbstractTensor wpe = weights.load("wpe.weight");

        return (inputToken, position) -> {
            AbstractTensor embedding = makeDenseTensor(1, c.embeddingLength);

            for (int i = 0; i < c.embeddingLength; i++) {
                float v = wte.get(inputToken, i) + wpe.get(position, i);
                embedding.set(v, 0, i);
            }

            return embedding;
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        TransformerBlock[] transformerBlocks = new TransformerBlock[c.dctx().numberOfLayers];

        for (int i = c.dctx().layerStart; i < c.dctx().layerEnd; i++) {
            String b = "h." + i + ".";
            String prefix = b + "attn.";

            AbstractTensor[] attnBias = weights.load(prefix + "c_attn.bias").split(3, 1);
            AbstractTensor[] attnWeights = weights.load(prefix + "c_attn.weight").transpose().split(3, 0);
            CausalSelfAttention attention = new CausalSelfAttention(
                this,
                i,
                attnBias[0],
                attnBias[1],
                attnBias[2],
                attnWeights[0],
                attnWeights[1],
                attnWeights[2],
                weights.load(prefix + "c_proj.bias"),
                weights.load(prefix + "c_proj.weight").transpose()
            );

            prefix = b + "mlp.";
            MLPBlock mlpBlock = new MLPBlock(
                this,
                c.activationFunction,
                weights.load(prefix + "c_fc.bias"),
                weights.load(prefix + "c_fc.weight").transpose(),
                weights.load(prefix + "c_proj.bias"),
                weights.load(prefix + "c_proj.weight").transpose()
            );

            LayerNorm layerNorm1 = new LayerNorm(this, weights.load(b + "ln_1.bias"), weights.load(b + "ln_1.weight"));
            LayerNorm layerNorm2 = new LayerNorm(this, weights.load(b + "ln_2.bias"), weights.load(b + "ln_2.weight"));

            transformerBlocks[i] = new TransformerBlock(this, i, layerNorm1, attention, layerNorm2, mlpBlock);
        }

        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        final AbstractTensor wte = weights.load("wte.weight");
        final LayerNorm layerNorm = new LayerNorm(this, weights.load("ln_f.bias"), weights.load("ln_f.weight"));

        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return layerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return wte;
            }
        };
    }
}
