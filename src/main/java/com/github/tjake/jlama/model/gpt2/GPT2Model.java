package com.github.tjake.jlama.model.gpt2;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.Tokenizer;
import com.github.tjake.jlama.safetensors.Weights;

import java.util.Optional;

public class GPT2Model extends AbstractModel {
    public final Tensor wte;
    public final Tensor wpe;

    public final LayerNorm layerNorm;

    public final TransformerBlock[] transformerBlocks;

    public GPT2Model(Config c, Weights w, Tokenizer tokenizer) {
        super(c, w, tokenizer);

        this.wte = w.load("wte.weight");
        this.wpe = w.load("wpe.weight");

        this.layerNorm = new LayerNorm(c, w.load("ln_f.bias"), w.load("ln_f.weight"));
        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        for (int i = 0; i < transformerBlocks.length; i++)
        {
            String b = "h." + i + ".";
            String prefix = b + "attn.";

            Tensor[] attnBias = w.load(prefix + "c_attn.bias").split(3, 0);
            Tensor[] attnWeights = w.load(prefix + "c_attn.weight").transpose().split(3, 0);
            CausalSelfAttention attention = new CausalSelfAttention(c, attnBias[0], attnBias[1], attnBias[2],
                    attnWeights[0], attnWeights[1], attnWeights[2],
                    w.load(prefix + "c_proj.bias"), w.load(prefix + "c_proj.weight").transpose(), Optional.empty());

            prefix = b + "mlp.";
            MLPBlock mlpBlock = new MLPBlock(c, ActivationFunction.Type.GELU,
                    w.load(prefix + "c_fc.bias"), w.load(prefix + "c_fc.weight").transpose(),
                    w.load(prefix + "c_proj.bias"), w.load( prefix + "c_proj.weight").transpose()
            );

            LayerNorm layerNorm1 = new LayerNorm(c, w.load(b + "ln_1.bias"), w.load(b + "ln_1.weight"));
            LayerNorm layerNorm2 = new LayerNorm(c, w.load(b + "ln_2.bias"), w.load(b + "ln_2.weight"));

            transformerBlocks[i] = new TransformerBlock(attention, layerNorm1, layerNorm2, mlpBlock);
        }
    }

    @Override
    protected Tensor inputTokenToEmbedding(int inputToken, int position) {
        Tensor embedding = c.bufferCache.get(c.embeddingLength);

        for (int i = 0; i < c.embeddingLength; i++) {
            float v = wte.get(inputToken, i) + wpe.get(position, i);
            embedding.set(v, i);
        }

        return embedding;
    }

    @Override
    protected TransformerBlock[] getTransformerBlocks() {
        return transformerBlocks;
    }

    @Override
    protected LayerNorm getOutputLayerNorm() {
        return layerNorm;
    }

    @Override
    protected Tensor getOutputLogitsWeights() {
        return wte;
    }

    public Tensor forward(int token_id, int pos, Tensor kvbuf) {

        Tensor embedding = c.bufferCache.get(c.embeddingLength);

        for (int i = 0; i < c.embeddingLength; i++) {
            float v = wte.get(token_id, i) + wpe.get(pos, i);
            embedding.set(v, i);
        }

        for (int i = 0; i < c.numberOfLayers; i++) {
            Tensor kvlayer = kvbuf.slice(i);
            Tensor ref = embedding; //reference so we can free
            embedding = transformerBlocks[i].forward(embedding, pos, kvlayer);
            ref.close();
        }
        return embedding;
    }
}
