package com.github.tjake.jlama.model.llama;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.WeightLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

public class LlamaModel extends AbstractModel{
    private static final Logger logger = LoggerFactory.getLogger(LlamaModel.class);

    private final Tensor wte;

    private final LayerNorm outputLayerNorm;

    private final Tensor noBias;

    private final TransformerBlock[] transformerBlocks;

    private final Tensor classificationWeights;

    public LlamaModel(Config config, WeightLoader weights, LlamaTokenizer tokenizer) {
        super(config, weights, tokenizer);

        logger.info("Loading model...");

        //LLama doesn't use bias, will optimize this away later
        this.noBias = new FloatBufferTensor(c.hiddenLength);

        this.wte = weights.load("model.embed_tokens.weight");
        this.outputLayerNorm = new RMSNorm(c, noBias, weights.load("model.norm.weight"));
        this.classificationWeights = weights.load("lm_head.weight");

        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(c.embeddingLength / c.numberOfHeads, c.contextLength, 10000.0 );

        for (int i = 0; i < c.numberOfLayers; i++) {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(c,
                    noBias, noBias, noBias,
                    weights.load(prefix + "q_proj.weight"),
                    weights.load(prefix + "k_proj.weight"),
                    weights.load(prefix + "v_proj.weight"),
                    noBias,
                    weights.load(prefix + "o_proj.weight"),
                    Optional.of(ropeFreqs));

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(c, ActivationFunction.Type.SILU,
                    noBias, weights.load(prefix + "gate_proj.weight"), //w1
                    noBias, weights.load(prefix + "down_proj.weight"), //w2
                    weights.load(prefix + "up_proj.weight"));          //w3

            this.transformerBlocks[i] = new TransformerBlock(attention,
                    new RMSNorm(c, noBias, weights.load(base + "input_layernorm.weight")),
                    new RMSNorm(c, noBias, weights.load(base + "post_attention_layernorm.weight")),
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
    protected Tensor getOutputLogitsWeights() {
        return classificationWeights;
    }

    @Override
    protected Tensor inputTokenToEmbedding(int inputToken, int position) {
        Tensor embedding = c.bufferCache.get(c.embeddingLength);

        VectorMath.pfor(0, c.embeddingLength, i -> {
            float v = wte.get(inputToken, i);
            embedding.set(v, i);
        });

        return embedding;
    }
}
