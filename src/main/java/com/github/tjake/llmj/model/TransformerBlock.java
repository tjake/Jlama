package com.github.tjake.llmj.model;

import com.github.tjake.llmj.math.VectorMath;

public class TransformerBlock {
    private final LayerNorm layerNorm1;
    private final CausalSelfAttention attention;
    private final LayerNorm layerNorm2;

    private final MLPBlock mlpBlock;

    public TransformerBlock(CausalSelfAttention attention, LayerNorm layerNorm1, LayerNorm layerNorm2, MLPBlock mlpBlock)
    {
        this.layerNorm1 = layerNorm1;
        this.attention = attention;

        this.layerNorm2 = layerNorm2;
        this.mlpBlock = mlpBlock;
    }

    public Tensor forward(Tensor embedding, int position, Tensor kvBuffer) {

        //emb += attn(ln_1(emb), kvbuf, i)
        Tensor lnemb = layerNorm1.forward(embedding);
        Tensor postAttention = attention.forward(lnemb, position, kvBuffer);

        //residual connection
        VectorMath.accumulate(postAttention, embedding);

        //emb += proj(gelu(fc(ln_2(emb))))
        Tensor lnemb2 = layerNorm2.forward(postAttention);
        Tensor postMlp = mlpBlock.forward(lnemb2);

        //residual connection
        VectorMath.accumulate(postMlp, postAttention);

        //Release any tmp buffers
        lnemb.close();
        lnemb2.close();
        postAttention.close();

        return postMlp;
    }
}
