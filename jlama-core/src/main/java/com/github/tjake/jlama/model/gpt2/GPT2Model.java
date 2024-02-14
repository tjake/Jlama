package com.github.tjake.jlama.model.gpt2;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.*;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.Optional;

public class GPT2Model extends AbstractModel {

    public GPT2Model(Config c, WeightLoader w, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(InferenceType.FULL_GENERATION, c, w, tokenizer, workingDType, workingQType, modelQType);
    }

    public GPT2Model(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        final AbstractTensor wte = weights.load("wte.weight");
        final AbstractTensor wpe = weights.load("wpe.weight");

        return (inputToken, position) -> {
            AbstractTensor embedding = makeTensor(c.embeddingLength);

            for (int i = 0; i < c.embeddingLength; i++) {
                float v = wte.get(inputToken, i) + wpe.get(position, i);
                embedding.set(v, i);
            }

            return embedding;
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        TransformerBlock[] transformerBlocks = new TransformerBlock[c.getNumberOfLayers()];

        for (int i = c.layerStart(); i < c.layerEnd(); i++)
        {
            String b = "h." + i + ".";
            String prefix = b + "attn.";

            AbstractTensor[] attnBias = weights.load(prefix + "c_attn.bias").split(3, 0);
            AbstractTensor[] attnWeights = weights.load(prefix + "c_attn.weight").transpose().split(3, 0);
            CausalSelfAttention attention = new CausalSelfAttention(this, attnBias[0], attnBias[1], attnBias[2],
                    attnWeights[0], attnWeights[1], attnWeights[2],
                    weights.load(prefix + "c_proj.bias"), weights.load(prefix + "c_proj.weight").transpose());

            prefix = b + "mlp.";
            MLPBlock mlpBlock = new MLPBlock(this, ActivationFunction.Type.GELU,
                    weights.load(prefix + "c_fc.bias"), weights.load(prefix + "c_fc.weight").transpose(),
                    weights.load(prefix + "c_proj.bias"), weights.load( prefix + "c_proj.weight").transpose()
            );

            LayerNorm layerNorm1 = new LayerNorm(this, weights.load(b + "ln_1.bias"), weights.load(b + "ln_1.weight"));
            LayerNorm layerNorm2 = new LayerNorm(this, weights.load(b + "ln_2.bias"), weights.load(b + "ln_2.weight"));

            transformerBlocks[i] = new TransformerBlock(this, layerNorm1, attention, layerNorm2, mlpBlock);
        }

        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        final AbstractTensor wte = weights.load("wte.weight");
        final LayerNorm layerNorm = new LayerNorm(this, weights.load("ln_f.bias"), weights.load("ln_f.weight"));

        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return layerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return wte;
            }
        };
    }
}
