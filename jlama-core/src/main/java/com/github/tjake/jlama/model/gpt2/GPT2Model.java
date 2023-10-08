package com.github.tjake.jlama.model.gpt2;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.Tokenizer;
import com.github.tjake.jlama.safetensors.Weights;
import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.Optional;

public class GPT2Model extends AbstractModel {
    private final AbstractTensor wte;
    private final AbstractTensor wpe;

    private final LayerNorm layerNorm;

    private final TransformerBlock[] transformerBlocks;

    public GPT2Model(Config c, Weights w, Tokenizer tokenizer, DType workingDType, DType workingQType) {
        super(c, w, tokenizer, workingDType, workingQType);

        this.wte = w.load("wte.weight");
        this.wpe = w.load("wpe.weight");

        this.layerNorm = new LayerNorm(this, w.load("ln_f.bias"), w.load("ln_f.weight"));
        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        for (int i = 0; i < transformerBlocks.length; i++)
        {
            String b = "h." + i + ".";
            String prefix = b + "attn.";

            AbstractTensor[] attnBias = w.load(prefix + "c_attn.bias").split(3, 0);
            AbstractTensor[] attnWeights = w.load(prefix + "c_attn.weight").transpose().split(3, 0);
            CausalSelfAttention attention = new CausalSelfAttention(this, attnBias[0], attnBias[1], attnBias[2],
                    attnWeights[0], attnWeights[1], attnWeights[2],
                    w.load(prefix + "c_proj.bias"), w.load(prefix + "c_proj.weight").transpose(), Optional.empty());

            prefix = b + "mlp.";
            MLPBlock mlpBlock = new MLPBlock(this, ActivationFunction.Type.GELU,
                    w.load(prefix + "c_fc.bias"), w.load(prefix + "c_fc.weight").transpose(),
                    w.load(prefix + "c_proj.bias"), w.load( prefix + "c_proj.weight").transpose()
            );

            LayerNorm layerNorm1 = new LayerNorm(this, w.load(b + "ln_1.bias"), w.load(b + "ln_1.weight"));
            LayerNorm layerNorm2 = new LayerNorm(this, w.load(b + "ln_2.bias"), w.load(b + "ln_2.weight"));

            transformerBlocks[i] = new TransformerBlock(this, layerNorm1, attention, layerNorm2, mlpBlock);
        }
    }

    @Override
    protected AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
        AbstractTensor embedding = makeTensor(c.embeddingLength);

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
    protected AbstractTensor getOutputLogitsWeights() {
        return wte;
    }

    @Override
    public AbstractTensor forward(int token_id, int pos, AbstractTensor kvbuf) {

        AbstractTensor embedding = makeTensor(c.embeddingLength);

        for (int i = 0; i < c.embeddingLength; i++) {
            float v = wte.get(token_id, i) + wpe.get(pos, i);
            embedding.set(v, i);
        }

        for (int i = 0; i < c.numberOfLayers; i++) {
            AbstractTensor kvlayer = kvbuf.slice(i);
            AbstractTensor ref = embedding; //reference so we can free
            embedding = transformerBlocks[i].forward(embedding, pos, kvlayer);
            ref.close();
        }
        return embedding;
    }
}
