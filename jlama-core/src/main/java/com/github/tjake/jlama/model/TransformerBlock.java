package com.github.tjake.jlama.model;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import java.util.Optional;

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

        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
        AbstractTensor postAttention;
        try(AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer);
        }

        //residual connection
        TensorOperationsProvider.get().accumulate(postAttention, embedding);

        AbstractTensor lnemb2 = postAttentionNorm.forward(postAttention);
        AbstractTensor postMlp;
        try(AbstractTensor qlnemb2 = model.maybeQuantize(lnemb2)) {
            postMlp = mlpBlock.forward(qlnemb2);
        }

        //residual connection
        TensorOperationsProvider.get().accumulate(postMlp, postAttention);

        //Release any tmp buffers
        if (lnemb != embedding)
            lnemb.close();

        lnemb2.close();
        postAttention.close();

        return postMlpNorm.map(ln -> {
            AbstractTensor lnout = ln.forward(postMlp);
            postMlp.close();
            return lnout;
        }).orElse(postMlp);
    }

    public AbstractTensor[] batchForward(AbstractTensor[] embeddings, int startPos, AbstractTensor kvBuffer, int batchSize) {
        boolean cleanlnemb = false;
        AbstractTensor[] lnemb = embeddings;
        if (preAttentionNorm.isPresent()) {
            cleanlnemb = true;
            LayerNorm ln = preAttentionNorm.get();
            lnemb = tmpArray.get();
            if (lnemb == null || lnemb.length < batchSize) {
                lnemb = new AbstractTensor[batchSize];
                tmpArray.set(lnemb);
            }
            for (int i = 0; i < batchSize; i++)
                lnemb[i] = ln.forward(embeddings[i]);
        }


        AbstractTensor[] postAttention = tmpArray2.get();
        if (postAttention == null || postAttention.length < batchSize) {
            postAttention = new AbstractTensor[batchSize];
            tmpArray2.set(postAttention);
        }

        for (int i = 0; i < batchSize; i++)
            postAttention[i] = attention.forward(lnemb[i], startPos + i, kvBuffer);

        //residual connection
        for (int i = 0; i < batchSize; i++)
            TensorOperationsProvider.get().accumulate(postAttention[i], embeddings[i]);

        //Release any tmp buffers
        if (cleanlnemb)
            for (int i = 0; i < batchSize; i++)
                lnemb[i].close();

        AbstractTensor[] lnemb2 = tmpArray.get();
        if (lnemb2 == null || lnemb2.length < batchSize) {
            lnemb2 = new AbstractTensor[batchSize];
            tmpArray.set(lnemb2);
        }
        for (int i = 0; i < batchSize; i++)
            lnemb2[i] = postAttentionNorm.forward(postAttention[i]);

        AbstractTensor[] postMlp = lnemb2;
        for (int i = 0; i < batchSize; i++) {
            AbstractTensor ref = lnemb2[i];
            postMlp[i] = mlpBlock.forward(ref);
            ref.close();
        }

        //residual connection
        for (int i = 0; i < batchSize; i++)
            TensorOperationsProvider.get().accumulate(postMlp[i], postAttention[i]);

        for (int i = 0; i < batchSize; i++)
            postAttention[i].close();


        if (postMlpNorm.isPresent()) {
            LayerNorm ln = postMlpNorm.get();
            for (int i = 0; i < batchSize; i++) {
                AbstractTensor ref = postMlp[i];
                postMlp[i] = ln.forward(ref);
                ref.close();
            }
        }

        return postMlp;
    }
}
