package com.github.tjake.jlama.model.llama;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import com.google.common.base.Preconditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

public class LlamaModel extends AbstractModel {
    private static final Logger logger = LoggerFactory.getLogger(LlamaModel.class);

    public LlamaModel(Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(InferenceType.FULL_GENERATION, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public LlamaModel(InferenceType inferenceType, Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    @Override
    protected EmbedInput loadInputWeights() {

        final AbstractTensor wte = weights.load("model.embed_tokens.weight", c.offset).quantize(workingDType); //Don't quantize this, it's used for the embedding layer

        return (inputToken, position) -> {
            AbstractTensor embedding = makeTensor(c.embeddingLength);
            embedding.copyFrom(wte, wte.getOffset(inputToken, c.embeddingSegmentStart()), embedding.getOffset(c.embeddingSegmentStart()), c.embeddingSegmentLength());
//            VectorMath.pfor(c.embeddingSegmentStart(), c.embeddingSegmentLength(), i -> {
//                float v = wte.get(inputToken, i);
//                embedding.set(v, i);
//            });

            return embedding;
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            logger.info("Quantizing model with {} - Please hold...", qType);
        }

        TransformerBlock[] transformerBlocks = new TransformerBlock[c.getNumberOfLayers()];

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(c.embeddingLength / c.numberOfHeads, c.contextLength, 10000.0 );

        IntStream.range(c.layerStart(), c.layerEnd()).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(this,
                    weights.load(prefix + "q_proj.weight", c.offset).quantize(qType),
                    weights.load(prefix + "k_proj.weight", c.offset).quantize(qType),
                    weights.load(prefix + "v_proj.weight", c.offset).quantize(qType),
                    weights.load(prefix + "o_proj.weight", c.offset).quantize(qType),
                    Optional.of(ropeFreqs));

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(this, ActivationFunction.Type.SILU,
                    weights.load(prefix + "gate_proj.weight", c.offset).quantize(qType), //w1
                    weights.load(prefix + "down_proj.weight").quantize(qType), //w2
                    weights.load(prefix + "up_proj.weight", c.offset).quantize(qType));  //w3

            transformerBlocks[i] = new TransformerBlock( this,
                    new RMSNorm(this, weights.load(base + "input_layernorm.weight", c.offset).quantize(qType)),
                    attention,
                    new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight", c.offset).quantize(qType)),
                    mlp);
        });

        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        final LayerNorm outputLayerNorm = new RMSNorm(this, weights.load("model.norm.weight").quantize(qType));
        final AbstractTensor classificationWeights = weights.load("lm_head.weight").quantize(workingDType); //Don't quantize this, it's the output layer

        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return outputLayerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return classificationWeights;
            }
        };
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
        Preconditions.checkArgument(t.dims() == 1 && t.shape().last() == c.embeddingLength, "Unexpected shape");
        if (t.dType() == workingQType)
            return super.maybeQuantize(t);

        return TensorOperationsProvider.get().quantize(t, workingQType, c.embeddingSegmentStart(), c.embeddingSegmentLength());
    }
}
