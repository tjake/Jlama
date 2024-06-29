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
package com.github.tjake.jlama.model.bert;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.util.Arrays;
import java.util.Optional;

public class BertModel extends AbstractModel {

    public BertModel(
            Config c,
            WeightLoader w,
            Tokenizer tokenizer,
            DType workingDType,
            DType workingQType,
            Optional<DType> modelQType) {
        super(InferenceType.FORWARD_PASS, c, w, tokenizer, workingDType, workingQType, modelQType);
    }

    public BertModel(
            InferenceType inferenceType,
            Config c,
            WeightLoader w,
            Tokenizer tokenizer,
            DType workingDType,
            DType workingQType,
            Optional<DType> modelQType) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        AbstractTensor we = weights.load("embeddings.word_embeddings.weight");
        AbstractTensor wte = weights.load("embeddings.token_type_embeddings.weight");
        AbstractTensor wpe = weights.load("embeddings.position_embeddings.weight");

        LayerNorm inputLayerNorm = new LayerNorm(
                this, weights.load("embeddings.LayerNorm.bias"), weights.load("embeddings.LayerNorm.weight"));

        return (inputToken, position) -> {
            AbstractTensor embedding = makeTensor(c.embeddingLength);

            for (int i = 0; i < c.embeddingLength; i++) {
                float v = we.get(inputToken, i) + wte.get(0, i) + wpe.get(position, i);
                embedding.set(v, 0, i);
            }

            AbstractTensor lnemb = inputLayerNorm.forward(embedding);
            embedding.close();
            return lnemb;
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        TransformerBlock[] transformerBlocks = new TransformerBlock[c.getNumberOfLayers()];

        for (int i = c.layerStart(); i < c.layerEnd(); i++) {
            String b = "encoder.layer." + i + ".";
            String prefix = b + "attention.";

            AbstractTensor keyBias = weights.load(prefix + "self.key.bias");
            AbstractTensor keyWeight = weights.load(prefix + "self.key.weight");

            AbstractTensor queryBias = weights.load(prefix + "self.query.bias");
            AbstractTensor queryWeight = weights.load(prefix + "self.query.weight");

            AbstractTensor valueBias = weights.load(prefix + "self.value.bias");
            AbstractTensor valueWeight = weights.load(prefix + "self.value.weight");

            AbstractTensor outputBias = weights.load(prefix + "output.dense.bias");
            AbstractTensor outputWeight = weights.load(prefix + "output.dense.weight");
            CausalSelfAttention attention = new CausalSelfAttention(
                    this, keyBias, queryBias, valueBias, keyWeight, queryWeight, valueWeight, outputBias, outputWeight);

            prefix = b;
            MLPBlock mlpBlock = new MLPBlock(
                    this,
                    ActivationFunction.Type.GELU,
                    weights.load(prefix + "intermediate.dense.bias"),
                    weights.load(prefix + "intermediate.dense.weight"),
                    weights.load(prefix + "output.dense.bias"),
                    weights.load(prefix + "output.dense.weight"));

            LayerNorm postAttentionNorm = new LayerNorm(
                    this,
                    weights.load(b + "attention.output.LayerNorm.bias"),
                    weights.load(b + "attention.output.LayerNorm.weight"));
            LayerNorm postMlpNorm = new LayerNorm(
                    this, weights.load(b + "output.LayerNorm.bias"), weights.load(b + "output.LayerNorm.weight"));

            transformerBlocks[i] = new TransformerBlock(this, attention, postAttentionNorm, mlpBlock, postMlpNorm);
        }

        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException();
    }

    public float[] embed(String input) {
        int[] encoded = Arrays.stream(tokenizer.encode(input))
                .mapToInt(Ints::checkedCast)
                .toArray();
        Preconditions.checkArgument(encoded.length < c.contextLength);
        float[] outputEmbedding = new float[c.embeddingLength];

        try (AbstractTensor kvmem =
                makeTensor(c.getNumberOfLayers(), 2, encoded.length, c.embeddingLength)) { // 2 for key and value

            int promptLength = encoded.length;
            float avgp = 1.0f / promptLength;

            AbstractTensor r = batchForward(encoded, 0, kvmem);
            for (int i = 0; i < promptLength; i++) {
                AbstractTensor output = r.slice(i);

                // Average Pooling
                for (int ii = 0; ii < c.embeddingLength; ii++) outputEmbedding[ii] += output.get(0, ii) * avgp;
            }
            r.close();
            VectorMath.l2normalize(outputEmbedding);
        }
        return outputEmbedding;
    }
}
