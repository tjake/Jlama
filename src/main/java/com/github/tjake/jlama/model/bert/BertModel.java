package com.github.tjake.jlama.model.bert;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.Tokenizer;
import com.github.tjake.jlama.safetensors.Weights;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;

public class BertModel extends AbstractModel {
    private final Tensor we;
    private final Tensor wte;
    private final Tensor wpe;

    private final LayerNorm inputLayerNorm;

    private final TransformerBlock[] transformerBlocks;

    public BertModel(Config c, Weights w, Tokenizer tokenizer) {
        super(c, w, tokenizer);

        this.we =  w.load("embeddings.word_embeddings.weight");
        this.wte = w.load("embeddings.token_type_embeddings.weight");
        this.wpe = w.load("embeddings.position_embeddings.weight");

        this.inputLayerNorm = new LayerNorm(c, w.load("embeddings.LayerNorm.bias"), w.load("embeddings.LayerNorm.weight"));

        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        for (int i = 0; i < c.numberOfLayers; i++) {
            String b = "encoder.layer." + i + ".";
            String prefix = b + "attention.";

            Tensor keyBias = w.load(prefix + "self.key.bias");
            Tensor keyWeight = w.load(prefix + "self.key.weight");

            Tensor queryBias = w.load(prefix + "self.query.bias");
            Tensor queryWeight = w.load(prefix + "self.query.weight");

            Tensor valueBias = w.load(prefix + "self.value.bias");
            Tensor valueWeight = w.load(prefix + "self.value.weight");

            Tensor outputBias = w.load(prefix + "output.dense.bias");
            Tensor outputWeight = w.load(prefix + "output.dense.weight");
            CausalSelfAttention attention = new CausalSelfAttention(c,
                    keyBias, queryBias, valueBias,
                    keyWeight, queryWeight, valueWeight,
                    outputBias, outputWeight, Optional.empty());

            prefix = b;
            MLPBlock mlpBlock = new MLPBlock(c, ActivationFunction.Type.GELU,
                    w.load(prefix + "intermediate.dense.bias"), w.load(prefix + "intermediate.dense.weight"),
                    w.load(prefix + "output.dense.bias"), w.load( prefix + "output.dense.weight")
            );

            LayerNorm postAttentionNorm = new LayerNorm(c, w.load(b + "attention.output.LayerNorm.bias"), w.load(b + "attention.output.LayerNorm.weight"));
            LayerNorm postMlpNorm = new LayerNorm(c, w.load(b + "output.LayerNorm.bias"), w.load(b + "output.LayerNorm.weight"));

            transformerBlocks[i] = new TransformerBlock(attention, postAttentionNorm, mlpBlock, postMlpNorm);
        }
    }

    @Override
    protected Tensor inputTokenToEmbedding(int inputToken, int position) {
        Tensor embedding = c.bufferCache.get(c.embeddingLength);

        for (int i = 0; i < c.embeddingLength; i++) {
            float v = we.get(inputToken, i) + wte.get(0, i) + wpe.get(position, i);
            embedding.set(v, i);
        }

        Tensor lnemb = this.inputLayerNorm.forward(embedding);
        embedding.close();
        return lnemb;
    }

    @Override
    protected TransformerBlock[] getTransformerBlocks() {
        return transformerBlocks;
    }

    @Override
    protected LayerNorm getOutputLayerNorm() {
        throw new IllegalStateException();
    }

    @Override
    protected Tensor getOutputLogitsWeights() {
        throw new IllegalStateException();
    }

    public float[] embed(String input) {
        long[] encoded = tokenizer.encode(input);
        Preconditions.checkArgument(encoded.length < c.contextLength);

        Tensor kvmem = c.bufferCache.get(c.numberOfLayers, encoded.length, c.embeddingLength * 2); //k and v are concatenated

        int promptLength = encoded.length;
        float avgp = 1.0f/promptLength;

        float[] outputEmbedding = new float[c.embeddingLength];

        for (int i = 0; i < promptLength; i++) {
            int next = (int)encoded[i];
            Tensor output = forward(next, i, kvmem);

            //Average Pooling
            for (int ii = 0; ii < c.embeddingLength; ii++)
                outputEmbedding[ii] += output.get(ii) * avgp;

            output.close();
        }

        VectorMath.l2normalize(outputEmbedding);
        kvmem.close();
        return outputEmbedding;
    }
}
