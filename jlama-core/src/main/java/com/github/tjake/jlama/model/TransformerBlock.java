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
package com.github.tjake.jlama.model;

import com.github.tjake.jlama.model.functions.FeedForward;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.Pair;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Consumer;

public class TransformerBlock {
    private final AbstractModel model;
    final Optional<LayerNorm> preAttentionNorm;
    final CausalSelfAttention attention;
    final LayerNorm postAttentionNorm;
    final FeedForward ffBlock;
    final Optional<LayerNorm> postFFNorm;

    public TransformerBlock(
            AbstractModel model,
            LayerNorm preAttentionNorm,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            FeedForward ffBlock) {
        this.model = model;
        this.preAttentionNorm = Optional.of(preAttentionNorm);
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.ffBlock = ffBlock;

        this.postFFNorm = Optional.empty();
    }

    public TransformerBlock(
            AbstractModel model,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm) {
        this.model = model;
        this.preAttentionNorm = Optional.empty();
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.ffBlock = ffBlock;

        this.postFFNorm = Optional.of(postFFNorm);
    }

    public AbstractTensor forward(AbstractTensor embedding, int position, AbstractTensor kvBuffer) {
        return forward(embedding, position, kvBuffer, Optional.empty(), Optional.empty());
    }

    public AbstractTensor forward(
            AbstractTensor embedding,
            int position,
            AbstractTensor kvBuffer,
            Optional<BiFunction<Float, Float, Pair<Float, Float>>> normReducer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {

        AbstractTensor lnemb =
                preAttentionNorm.map(ln -> ln.forward(embedding, normReducer)).orElse(embedding);
        AbstractTensor postAttention;
        try (AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, tensorReducer);
        }

        // residual connection
        TensorOperationsProvider.get()
                .accumulate(
                        postAttention, embedding, model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength());

        AbstractTensor lnemb2 = postAttentionNorm.forward(postAttention, normReducer);
        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnemb2)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer);
        }

        // residual connection
        TensorOperationsProvider.get()
                .accumulate(postFF, postAttention, model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength());

        // Release any tmp buffers
        if (lnemb != embedding) lnemb.close();

        lnemb2.close();
        postAttention.close();

        return postFFNorm
                .map(ln -> {
                    AbstractTensor lnout = ln.forward(postFF, normReducer);
                    postFF.close();
                    return lnout;
                })
                .orElse(postFF);
    }
}
