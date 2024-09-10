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
package com.github.tjake.jlama.model.llama;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.util.Optional;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LlamaModel extends AbstractModel {
    private static final Logger logger = LoggerFactory.getLogger(LlamaModel.class);

    public LlamaModel(
        Config config,
        WeightLoader weights,
        Tokenizer tokenizer,
        DType workingDType,
        DType workingQType,
        Optional<DType> modelQType
    ) {
        super(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public LlamaModel(
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
        return ModelSupport.ModelType.LLAMA;
    }

    @Override
    protected EmbedInput loadInputWeights() {

        // Don't quantize this, it's used for the embedding layer
        final AbstractTensor wte = weights.load("model.embed_tokens.weight").quantize(workingDType);

        return (inputToken, position) -> {
            AbstractTensor embedding = makeDenseTensor(1, c.embeddingLength);

            AbstractTensor at = wte.slice(true, inputToken);

            if (wte.dType() != embedding.dType())
                at = TensorOperationsProvider.get().quantize(at, embedding.dType(), 0, c.embeddingLength);

            //Always copy the entire embedding
            embedding.copyFrom(at, 0, 0, c.embeddingLength);

            return embedding;
        };
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
                weights.load(prefix + "up_proj.weight", c.dctx(), true, false).quantize(qType)
            ); // w3

            transformerBlocks[i] = new TransformerBlock(
                this,
                i,
                new RMSNorm(this, weights.load(base + "input_layernorm.weight").quantize(qType)),
                attention,
                new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight").quantize(qType)),
                mlp
            );
        });

        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        final LayerNorm outputLayerNorm = new RMSNorm(this, weights.load("model.norm.weight").quantize(qType));
        final AbstractTensor classificationWeights = weights.load("lm_head.weight").quantize(workingDType); // Don't quantize this, it's the
                                                                                                            // output layer

        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return outputLayerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return classificationWeights;
            }
        };
    }

    @Override
    protected AbstractTensor maybeQuantize(AbstractTensor t) {
        Preconditions.checkArgument(t.dims() == 2, "Unexpected shape");
        if (t.dType() == workingQType) return super.maybeQuantize(t);

        return //t.shape().last() == c.embeddingLength
            //? TensorOperationsProvider.get().quantize(t, workingQType, c.embeddingSegmentStart(), c.embeddingSegmentLength())
            //:
        TensorOperationsProvider.get().quantize(t, workingQType, 0, Ints.checkedCast(t.shape().last()));
    }
}
