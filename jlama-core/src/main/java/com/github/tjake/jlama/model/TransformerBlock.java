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
    final Optional<LayerNorm> postAttentionNorm; // After attention, before the residual connection
    final Optional<LayerNorm> preFFNorm; // After residual connection, before the FF
    final FeedForward ffBlock;
    final Optional<LayerNorm> postFFNorm; // After FF, before the residual connection
    final Optional<LayerNorm> preResponseNorm; // After the residual connection

    public TransformerBlock(
        AbstractModel model,
        int layerIndex,
        LayerNorm preAttentionNorm,
        CausalSelfAttention attention,
        LayerNorm postAttentionNorm,
        FeedForward ffBlock
    ) {
        this(
            model,
            layerIndex,
            Optional.of(preAttentionNorm),
            attention,
            Optional.empty(),
            Optional.of(postAttentionNorm),
            ffBlock,
            Optional.empty(),
            Optional.empty()
        );
    }

    public TransformerBlock(
        AbstractModel model,
        int layerIndex,
        CausalSelfAttention attention,
        LayerNorm postAttentionNorm,
        FeedForward ffBlock,
        LayerNorm postFFNorm
    ) {
        this(
            model,
            layerIndex,
            Optional.empty(),
            attention,
            Optional.empty(),
            Optional.of(postAttentionNorm),
            ffBlock,
            Optional.empty(),
            Optional.of(postFFNorm)
        );
    }

    public TransformerBlock(
        AbstractModel model,
        int layerIndex,
        LayerNorm preAttentionNorm,
        CausalSelfAttention attention,
        LayerNorm postAttentionNorm,
        FeedForward ffBlock,
        LayerNorm postFFNorm
    ) {
        this(
            model,
            layerIndex,
            Optional.of(preAttentionNorm),
            attention,
            Optional.empty(),
            Optional.of(postAttentionNorm),
            ffBlock,
            Optional.empty(),
            Optional.of(postFFNorm)
        );
    }

    public TransformerBlock(
        AbstractModel model,
        int layerIndex,
        LayerNorm preAttentionNorm,
        CausalSelfAttention attention,
        LayerNorm postAttentionNorm,
        LayerNorm preFFNorm,
        FeedForward ffBlock,
        LayerNorm postFFNorm
    ) {
        this(
            model,
            layerIndex,
            Optional.of(preAttentionNorm),
            attention,
            Optional.of(postAttentionNorm),
            Optional.of(preFFNorm),
            ffBlock,
            Optional.of(postFFNorm),
            Optional.empty()
        );
    }

    public TransformerBlock(
        AbstractModel model,
        int layerIndex,
        Optional<LayerNorm> preAttentionNorm,
        CausalSelfAttention attention,
        Optional<LayerNorm> postAttentionNorm,
        Optional<LayerNorm> preFFNorm,
        FeedForward ffBlock,
        Optional<LayerNorm> postFFNorm,
        Optional<LayerNorm> preResponseNorm
    ) {

        this.model = model;
        this.layerIndex = layerIndex;
        this.preAttentionNorm = preAttentionNorm;
        this.attention = attention;
        this.postAttentionNorm = postAttentionNorm;
        this.preFFNorm = preFFNorm;
        this.ffBlock = ffBlock;
        this.postFFNorm = postFFNorm;
        this.preResponseNorm = preResponseNorm;
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
        AbstractTensor lnattn = maybeApplyNorm(postAttention, postAttentionNorm);

        debug("post_attn_norm", lnattn, layerIndex);

        // residual connection
        if (model.c.residualMultiplier != null) {
            TensorOperationsProvider.get().scale(model.c.residualMultiplier, lnattn, 0, model.c.embeddingLength);
        }
        TensorOperationsProvider.get().accumulate(lnattn, embedding, 0, model.c.embeddingLength);

        AbstractTensor lnpreFF = preFFNorm.map(ln -> ln.forward(lnattn)).orElse(lnattn);

        debug("pre_ff_norm", lnpreFF, layerIndex);

        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnpreFF)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer);
            debug("post_ff", postFF, layerIndex);
        }

        AbstractTensor lnpostFF = maybeApplyNorm(postFF, postFFNorm);

        // residual connection
        if (model.c.residualMultiplier != null) {
            TensorOperationsProvider.get().scale(model.c.residualMultiplier, lnpostFF, 0, model.c.embeddingLength);
        }
        TensorOperationsProvider.get().accumulate(lnpostFF, lnattn, 0, model.c.embeddingLength);

        debug("post_ff_res", lnpostFF, layerIndex);

        // Release any tmp buffers (embedding is released by caller)
        if (lnemb != embedding) lnemb.close();
        if (lnattn != postAttention) lnattn.close();
        else postAttention.close();
        if (lnpreFF != lnattn) lnpreFF.close();
        else lnattn.close();

        return maybeApplyNorm(lnpostFF, preResponseNorm);
    }

    private AbstractTensor maybeApplyNorm(AbstractTensor tensor, Optional<LayerNorm> norm) {
        return norm.map(ln -> {
            AbstractTensor o = ln.forward(tensor);
            tensor.close();
            return o;
        }).orElse(tensor);
    }
}
