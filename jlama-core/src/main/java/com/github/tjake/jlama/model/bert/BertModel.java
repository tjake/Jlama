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

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.model.functions.ClassifyOutput;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.PoolingLayer;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Optional;

public class BertModel extends AbstractModel {

    private static final String[] prefixes = new String[] { "", "bert." };

    public BertModel(Config c, WeightLoader w, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(InferenceType.FORWARD_PASS, c, w, tokenizer, workingDType, workingQType, modelQType);
    }

    public BertModel(
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

    protected AbstractTensor loadWeight(String name) {

        for (String prefix : prefixes) {
            String key = prefix + name;
            if (weights.isWeightPresent(key)) {
                return weights.load(key);
            }
        }

        throw new NoSuchElementException(Arrays.toString(prefixes) + " " + name + " not found in weights");
    }

    @Override
    public ModelSupport.ModelType getModelType() {
        return ModelSupport.getModelType("BERT");
    }

    @Override
    protected EmbedInput loadInputWeights() {
        AbstractTensor we = loadWeight("embeddings.word_embeddings.weight");
        AbstractTensor wte = loadWeight("embeddings.token_type_embeddings.weight");
        AbstractTensor wpe = loadWeight("embeddings.position_embeddings.weight");

        LayerNorm inputLayerNorm = new LayerNorm(this, loadWeight("embeddings.LayerNorm.bias"), loadWeight("embeddings.LayerNorm.weight"));

        return (inputToken, position) -> {
            AbstractTensor embedding = makeDenseTensor(c.embeddingLength);

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
        TransformerBlock[] transformerBlocks = new TransformerBlock[c.dctx().embeddingSegmentLength];

        for (int i = c.dctx().layerStart; i < c.dctx().layerEnd; i++) {
            String b = "encoder.layer." + i + ".";
            String prefix = b + "attention.";

            AbstractTensor keyBias = loadWeight(prefix + "self.key.bias");
            AbstractTensor keyWeight = loadWeight(prefix + "self.key.weight");

            AbstractTensor queryBias = loadWeight(prefix + "self.query.bias");
            AbstractTensor queryWeight = loadWeight(prefix + "self.query.weight");

            AbstractTensor valueBias = loadWeight(prefix + "self.value.bias");
            AbstractTensor valueWeight = loadWeight(prefix + "self.value.weight");

            AbstractTensor outputBias = loadWeight(prefix + "output.dense.bias");
            AbstractTensor outputWeight = loadWeight(prefix + "output.dense.weight");
            CausalSelfAttention attention = new CausalSelfAttention(
                this,
                i,
                keyBias,
                queryBias,
                valueBias,
                keyWeight,
                queryWeight,
                valueWeight,
                outputBias,
                outputWeight
            );

            prefix = b;
            MLPBlock mlpBlock = new MLPBlock(
                this,
                c.activationFunction,
                loadWeight(prefix + "intermediate.dense.bias"),
                loadWeight(prefix + "intermediate.dense.weight"),
                loadWeight(prefix + "output.dense.bias"),
                loadWeight(prefix + "output.dense.weight")
            );

            LayerNorm postAttentionNorm = new LayerNorm(
                this,
                loadWeight(b + "attention.output.LayerNorm.bias"),
                loadWeight(b + "attention.output.LayerNorm.weight")
            );
            LayerNorm postMlpNorm = new LayerNorm(this, loadWeight(b + "output.LayerNorm.bias"), loadWeight(b + "output.LayerNorm.weight"));

            transformerBlocks[i] = new TransformerBlock(this, i, attention, postAttentionNorm, mlpBlock, postMlpNorm);
        }

        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected PoolingLayer loadPoolingWeights() {

        final AbstractTensor poolerDenseWeight = loadWeight("pooler.dense.weight");
        final AbstractTensor poolerDenseBias = loadWeight("pooler.dense.bias");

        return new PoolingLayer() {
            public AbstractTensor getPoolingWeights() {
                return poolerDenseWeight;
            }

            public Optional<AbstractTensor> getPoolingBias() {
                return Optional.of(poolerDenseBias);
            }
        };
    }

    @Override
    protected ClassifyOutput loadClassifierWeights() {
        if (c.isClassifier()) {
            final AbstractTensor classifierWeight = loadWeight("classifier.weight");
            final AbstractTensor classifierBias = loadWeight("classifier.bias");

            return new ClassifyOutput() {
                @Override
                public AbstractTensor getClassificationWeights() {
                    return classifierWeight;
                }

                @Override
                public Optional<AbstractTensor> getClassificationBias() {
                    return Optional.of(classifierBias);
                }
            };
        } else {
            throw new UnsupportedOperationException("Classification not supported by this model");
        }
    }
}
