package com.github.tjake.jlama.model;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.Pair;

import java.util.Optional;
import java.util.UUID;
import java.util.function.BiFunction;

public class TransformerBlock {
    private final AbstractModel model;
    private final Optional<LayerNorm> preAttentionNorm;
    private final CausalSelfAttention attention;
    private final LayerNorm postAttentionNorm;

    private final MLPBlock mlpBlock;

    private final Optional<LayerNorm> postMlpNorm;
    private static final ThreadLocal<AbstractTensor[]> tmpArray = new ThreadLocal<>();
    private static final ThreadLocal<AbstractTensor[]> tmpArray2 = new ThreadLocal<>();

    public TransformerBlock(AbstractModel model, LayerNorm preAttentionNorm, CausalSelfAttention attention, LayerNorm postAttentionNorm, MLPBlock mlpBlock)
    {
        this.model = model;
        this.preAttentionNorm = Optional.of(preAttentionNorm);
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.mlpBlock = mlpBlock;

        this.postMlpNorm = Optional.empty();
    }

    public TransformerBlock(AbstractModel model, CausalSelfAttention attention, LayerNorm postAttentionNorm, MLPBlock mlpBlock, LayerNorm postMlpNorm)
    {
        this.model = model;
        this.preAttentionNorm = Optional.empty();
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.mlpBlock = mlpBlock;

        this.postMlpNorm = Optional.of(postMlpNorm);
    }

    public AbstractTensor forward(AbstractTensor embedding, int position, AbstractTensor kvBuffer) {
        return forward(embedding, position, kvBuffer, Optional.empty());
    }

    public AbstractTensor forward(AbstractTensor embedding, int position, AbstractTensor kvBuffer, Optional<BiFunction<Float, Float, Pair<Float, Float>>> reducer) {

        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding, reducer)).orElse(embedding);
        AbstractTensor postAttention;
        try(AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, reducer);
        }

        //residual connection
        TensorOperationsProvider.get().accumulate(postAttention, embedding, model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength());

        AbstractTensor lnemb2 = postAttentionNorm.forward(postAttention, reducer);
        AbstractTensor postMlp;
        try(AbstractTensor qlnemb2 = model.maybeQuantize(lnemb2)) {
            postMlp = mlpBlock.forward(qlnemb2);
        }

        //residual connection
        TensorOperationsProvider.get().accumulate(postMlp, postAttention, model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength());

        //Release any tmp buffers
        if (lnemb != embedding)
            lnemb.close();

        lnemb2.close();
        postAttention.close();

        return postMlpNorm.map(ln -> {
            AbstractTensor lnout = ln.forward(postMlp, reducer);
            postMlp.close();
            return lnout;
        }).orElse(postMlp);
    }
}
