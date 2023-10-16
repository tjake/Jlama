package com.github.tjake.jlama.model.llama;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.Tokenizer;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import org.checkerframework.common.value.qual.IntRange;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

public class LlamaModel extends AbstractModel {
    private static final Logger logger = LoggerFactory.getLogger(LlamaModel.class);

    private final AbstractTensor wte;

    private final LayerNorm outputLayerNorm;

    private final TransformerBlock[] transformerBlocks;

    private final AbstractTensor classificationWeights;

    public LlamaModel(Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType) {
        super(config, weights, tokenizer, workingDType, workingQType);

        DType qType = DType.Q4;

        logger.info("Quantizing model with {} - Please hold...", qType);


        this.wte = weights.load("model.embed_tokens.weight").quantize(workingDType); //Don't quantize this, it's used for the embedding layer
        this.outputLayerNorm = new RMSNorm(this, weights.load("model.norm.weight").quantize(qType));
        this.classificationWeights = weights.load("lm_head.weight").quantize(workingDType); //Don't quantize this, it's the output layer

        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(c.embeddingLength / c.numberOfHeads, c.contextLength, 10000.0 );

        IntStream.range(0, c.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(this,
                    weights.load(prefix + "q_proj.weight").quantize(qType),
                    weights.load(prefix + "k_proj.weight").quantize(qType),
                    weights.load(prefix + "v_proj.weight").quantize(qType),
                    weights.load(prefix + "o_proj.weight").quantize(qType),
                    Optional.of(ropeFreqs));

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(this, ActivationFunction.Type.SILU,
                    weights.load(prefix + "gate_proj.weight").quantize(qType), //w1
                    weights.load(prefix + "down_proj.weight").quantize(qType), //w2
                    weights.load(prefix + "up_proj.weight").quantize(qType));  //w3

            this.transformerBlocks[i] = new TransformerBlock( this,
                    new RMSNorm(this, weights.load(base + "input_layernorm.weight").quantize(qType)), attention,
                    new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight").quantize(qType)),
                    mlp);
        });
        logger.info("Model loaded!");
    }

    @Override
    public String wrapPrompt(String prompt, Optional<String> systemPrompt)
    {
        StringBuilder b = new StringBuilder();
        b.append("[INST] ");
        if (systemPrompt.isPresent()) {
            b.append("<<SYS>> \n")
                    .append(systemPrompt.get())
                    .append("\n<</SYS>> \n\n");
        }
        b.append(prompt)
                .append(" [/INST]");

        return b.toString();
    }

    @Override
    protected AbstractTensor maybeQuantize(AbstractTensor t)
    {
        if (t.dType() == workingQType)
            return super.maybeQuantize(t);

        return TensorOperationsProvider.get().quantize(t, workingQType);
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
