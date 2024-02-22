package com.github.tjake.jlama.model.gemma;

import java.util.Optional;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.tjake.jlama.model.CausalSelfAttention;
import com.github.tjake.jlama.model.LayerNorm;
import com.github.tjake.jlama.model.MLPBlock;
import com.github.tjake.jlama.model.RMSNorm;
import com.github.tjake.jlama.model.TransformerBlock;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

public class GemmaModel extends LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(GemmaModel.class);

    public GemmaModel(Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType)
    {
        super(config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public GemmaModel(InferenceType inferenceType, Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType)
    {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    private AbstractTensor wte;

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            logger.info("Quantizing model with {} - Please hold...", qType);
        }

        TransformerBlock[] transformerBlocks = new TransformerBlock[c.getNumberOfLayers()];

        IntStream.range(c.layerStart(), c.layerEnd()).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(this,
                    weights.load(prefix + "q_proj.weight", c.offset()).quantize(qType),
                    weights.load(prefix + "k_proj.weight", c.offset()).quantize(qType),
                    weights.load(prefix + "v_proj.weight", c.offset()).quantize(qType),
                    weights.load(prefix + "o_proj.weight", c.offset()).quantize(qType));

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(this, c.activationFunction,
                    weights.load(prefix + "gate_proj.weight", c.offset()).quantize(qType), //w1
                    weights.load(prefix + "down_proj.weight").quantize(qType), //w2
                    weights.load(prefix + "up_proj.weight", c.offset()).quantize(qType));  //w3

            transformerBlocks[i] = new TransformerBlock( this,
                    new RMSNorm(this, weights.load(base + "input_layernorm.weight", c.offset()).quantize(qType), 1.0f),
                    attention,
                    new RMSNorm(this, weights.load(base + "post_attention_layernorm.weight", c.offset()).quantize(qType), 1.0f),
                    mlp);
        });

        return transformerBlocks;
    }

    @Override
    protected EmbedInput loadInputWeights() {

        if (wte == null)
            wte = weights.load("model.embed_tokens.weight", c.offset()).quantize(workingDType); //Don't quantize this, it's used for the embedding layer

        return (inputToken, position) -> {
            AbstractTensor embedding = makeTensor(c.embeddingLength);
            embedding.copyFrom(wte, wte.getOffset(inputToken, c.embeddingSegmentStart()), embedding.getOffset(c.embeddingSegmentStart()), c.embeddingSegmentLength());

            //This is important for Gemma, but not for Llama
            TensorOperationsProvider.get().scale((float) Math.pow(c.embeddingLength, 0.5), embedding, c.embeddingSegmentStart(), c.embeddingSegmentLength());

            return embedding;
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);

        if (wte == null)
            wte = weights.load("model.embed_tokens.weight", c.offset()).quantize(workingDType); //Don't quantize this, it's used for the embedding layer

        final LayerNorm layerNorm = new RMSNorm(this, weights.load("model.norm.weight").quantize(qType), 1.0f);

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
