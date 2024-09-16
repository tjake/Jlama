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

import static com.github.tjake.jlama.util.DebugSupport.debug;

import com.github.tjake.jlama.model.functions.FeedForward;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TransformerBlock {

    private static final Logger logger = LoggerFactory.getLogger(TransformerBlock.class);

    private final AbstractModel model;
    final int layerIndex;
    final Optional<LayerNorm> preAttentionNorm;
    final CausalSelfAttention attention;
    final LayerNorm postAttentionNorm;
    final FeedForward ffBlock;
    final Optional<LayerNorm> postFFNorm;

    public TransformerBlock(
        AbstractModel model,
        int layerIndex,
        LayerNorm preAttentionNorm,
        CausalSelfAttention attention,
        LayerNorm postAttentionNorm,
        FeedForward ffBlock
    ) {
        this.model = model;
        this.layerIndex = layerIndex;
        this.preAttentionNorm = Optional.of(preAttentionNorm);
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.ffBlock = ffBlock;

        this.postFFNorm = Optional.empty();
    }

    public TransformerBlock(
        AbstractModel model,
        int layerIndex,
        CausalSelfAttention attention,
        LayerNorm postAttentionNorm,
        FeedForward ffBlock,
        LayerNorm postFFNorm
    ) {
        this.model = model;
        this.layerIndex = layerIndex;
        this.preAttentionNorm = Optional.empty();
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.ffBlock = ffBlock;

        this.postFFNorm = Optional.of(postFFNorm);
    }

    public AbstractTensor forward(AbstractTensor embedding, int position, KvBufferCache.KvBuffer kvBuffer) {
        return forward(embedding, position, kvBuffer, Optional.empty());
    }

    public AbstractTensor forward(
        AbstractTensor embedding,
        int position,
        KvBufferCache.KvBuffer kvBuffer,
        Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {

        debug("input_emb", embedding, layerIndex);

        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);

        debug("ln_emb", lnemb, layerIndex);

        AbstractTensor postAttention;
        try (AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, tensorReducer);
        }

        debug("post_attn", postAttention, layerIndex);

        // residual connection
        TensorOperationsProvider.get().accumulate(postAttention, embedding, 0, model.c.embeddingLength);

        debug("post_attn_res", postAttention, layerIndex);

        AbstractTensor lnemb2 = postAttentionNorm.forward(postAttention);

        debug("ln_emb2", lnemb2, layerIndex);

        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnemb2)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer);
            debug("post_ff", postFF, layerIndex);
        }

        // residual connection
        TensorOperationsProvider.get().accumulate(postFF, postAttention, 0, model.c.embeddingLength);

        debug("post_ff_res", postFF, layerIndex);

        // Release any tmp buffers
        if (lnemb != embedding) lnemb.close();

        lnemb2.close();
        postAttention.close();

        return postFFNorm.map(ln -> {
            AbstractTensor lnout = ln.forward(postFF);
            debug("ln_out", lnout, layerIndex);
            postFF.close();
            return lnout;
        }).orElse(postFF);
    }
}
