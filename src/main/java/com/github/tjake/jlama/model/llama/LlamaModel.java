package com.github.tjake.jlama.model.llama;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

public class LlamaModel extends AbstractModel {
    private static final Logger logger = LoggerFactory.getLogger(LlamaModel.class);

    private final AbstractTensor wte;

    private final LayerNorm outputLayerNorm;

    private final AbstractTensor noBias;

    private final TransformerBlock[] transformerBlocks;

    private final AbstractTensor classificationWeights;

    public LlamaModel(Config config, WeightLoader weights, LlamaTokenizer tokenizer) {
        super(config, weights, tokenizer);

        logger.info("Loading model...");

        //LLama doesn't use bias, will optimize this away later
        this.noBias = new FloatBufferTensor(c.hiddenLength);

        this.wte = weights.load("model.embed_tokens.weight").quantize(DType.F32); //Don't quantize this, it's used for the embedding layer
        this.outputLayerNorm = new RMSNorm(this, noBias, weights.load("model.norm.weight").quantize(DType.I8));
        this.classificationWeights = weights.load("lm_head.weight").quantize(DType.F32); //Don't quantize this, it's the output layer

        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(c.embeddingLength / c.numberOfHeads, c.contextLength, 10000.0 );

        for (int i = 0; i < c.numberOfLayers; i++) {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(this,
                    noBias, noBias, noBias,
                    weights.load(prefix + "q_proj.weight").quantize(DType.I8),
                    weights.load(prefix + "k_proj.weight").quantize(DType.I8),
                    weights.load(prefix + "v_proj.weight").quantize(DType.I8),
                    noBias,
                    weights.load(prefix + "o_proj.weight").quantize(DType.I8),
                    Optional.of(ropeFreqs));

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(this, ActivationFunction.Type.SILU,
                    noBias, weights.load(prefix + "gate_proj.weight").quantize(DType.I8), //w1
                    noBias, weights.load(prefix + "down_proj.weight").quantize(DType.I8), //w2
                    weights.load(prefix + "up_proj.weight").quantize(DType.I8));          //w3

            this.transformerBlocks[i] = new TransformerBlock(new RMSNorm(this, noBias, weights.load(base + "input_layernorm.weight").quantize(DType.I8)), attention,
                    new RMSNorm(this, noBias, weights.load(base + "post_attention_layernorm.weight").quantize(DType.I8)),
                    mlp);
        }
        logger.info("Model loaded!");
    }

    @Override
    protected TransformerBlock[] getTransformerBlocks() {
        return transformerBlocks;
    }

    @Override
    protected LayerNorm getOutputLayerNorm() {
        return outputLayerNorm;
    }

    @Override
    protected AbstractTensor getOutputLogitsWeights() {
        return classificationWeights;
    }

    @Override
    protected AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
        AbstractTensor embedding = makeTensor(c.embeddingLength);

        VectorMath.pfor(0, c.embeddingLength, i -> {
            float v = wte.get(inputToken, i);
            embedding.set(v, i);
        });

        return embedding;
    }
}
