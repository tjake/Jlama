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
package com.github.tjake.jlama.model.gemma2;

import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

public class Gemma2Model extends LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(Gemma2Model.class);

    private final float embeddingScalingFactor;
    private AbstractTensor wte;

    public Gemma2Model(
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        this(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public Gemma2Model(
        InferenceType inferenceType,
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
        // https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/models/gemma/modeling_gemma.py#L898
        // This is the scaling factor for the embedding layer but google's implementation is a is rounded to 16 bits
        this.embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(
            FloatConversions.float32ToBFloat16((float) Math.pow(c.embeddingLength, 0.5))
        );
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("GEMMA2");
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
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
                weights.load(prefix + "o_proj.weight", c.dctx(), false, true).quantize(qType)
            );

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(
                this,
                c.activationFunction,
                weights.load(prefix + "gate_proj.weight", c.dctx(), true, false).quantize(qType), // w1
                weights.load(prefix + "down_proj.weight", c.dctx(), false, true).quantize(qType), // w2
                weights.load(prefix + "up_proj.weight", c.dctx(), true, false).quantize(qType) // w3
            );

            transformerBlocks[i] = new TransformerBlock(
                this,
                i,
                new RMSNorm(this, weights.load(base + "input_layernorm.weight").quantize(qType), 1.0f),
                attention,
                new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight").quantize(qType), 1.0f),
                new RMSNorm(this, weights.load(base + "pre_feedforward_layernorm.weight").quantize(qType), 1.0f),
                mlp,
                new RMSNorm(this, weights.load(base + "post_feedforward_layernorm.weight").quantize(qType), 1.0f)
            );
        });

        return transformerBlocks;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        // Don't quantize this, it's used for the embedding layer
        if (wte == null) wte = weights.load("model.embed_tokens.weight").quantize(workingDType);

        return (inputToken, position) -> {
            AbstractTensor embedding = makeDenseTensor(c.embeddingLength);
            AbstractTensor at = wte.slice(true, inputToken);
            if (wte.dType() != embedding.dType()) at = TensorOperationsProvider.get().quantize(at, embedding.dType(), 0, c.embeddingLength);

            embedding.copyFrom(at, 0, 0, c.embeddingLength);

            // This is important for Gemma, but not for Llama
            TensorOperationsProvider.get().scale(embeddingScalingFactor, embedding, 0, c.embeddingLength);

            return embedding;
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);

        if (wte == null) wte = weights.load("model.embed_tokens.weight").quantize(workingDType);

        final LayerNorm layerNorm = new RMSNorm(this, weights.load("model.norm.weight").quantize(qType), 1.0f);

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
