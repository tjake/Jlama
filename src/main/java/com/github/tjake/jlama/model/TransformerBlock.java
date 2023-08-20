package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;

import java.util.Optional;

public class TransformerBlock {
    private final Optional<LayerNorm> preAttentionNorm;
    private final CausalSelfAttention attention;
    private final LayerNorm postAttentionNorm;

    private final MLPBlock mlpBlock;

    private final Optional<LayerNorm> postMlpNorm;

    public TransformerBlock(LayerNorm preAttentionNorm, CausalSelfAttention attention, LayerNorm postAttentionNorm, MLPBlock mlpBlock)
    {
        this.preAttentionNorm = Optional.of(preAttentionNorm);
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.mlpBlock = mlpBlock;

        this.postMlpNorm = Optional.empty();
    }

    public TransformerBlock(CausalSelfAttention attention, LayerNorm postAttentionNorm, MLPBlock mlpBlock, LayerNorm postMlpNorm)
    {
        this.preAttentionNorm = Optional.empty();
        this.attention = attention;

        this.postAttentionNorm = postAttentionNorm;
        this.mlpBlock = mlpBlock;

        this.postMlpNorm = Optional.of(postMlpNorm);
    }

    public Tensor forward(Tensor embedding, int position, Tensor kvBuffer) {

        Tensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
        Tensor postAttention = attention.forward(lnemb, position, kvBuffer);
        //residual connection
        VectorMath.accumulate(postAttention, embedding);

        Tensor lnemb2 = postAttentionNorm.forward(postAttention);
        Tensor postMlp = mlpBlock.forward(lnemb2);
        //residual connection
        VectorMath.accumulate(postMlp, postAttention);

        //Release any tmp buffers
        if (lnemb != embedding)
            lnemb.close();

        lnemb2.close();
        postAttention.close();

        Tensor output = postMlpNorm.map(ln -> {
            Tensor lnout = ln.forward(postMlp);
            postMlp.close();
            return lnout;
        }).orElse(postMlp);

        return output;
    }
}
