package com.github.tjake.jlama.model.bert;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.Tokenizer;
import com.github.tjake.jlama.safetensors.Weights;
import com.github.tjake.jlama.tensor.AbstractTensor;

import com.google.common.base.Preconditions;

import java.util.Optional;

public class BertModel extends AbstractModel {
    private final AbstractTensor we;
    private final AbstractTensor wte;
    private final AbstractTensor wpe;

    private final LayerNorm inputLayerNorm;

    private final TransformerBlock[] transformerBlocks;

    public BertModel(Config c, Weights w, Tokenizer tokenizer, DType workingDType, DType workingQType) {
        super(c, w, tokenizer, workingDType, workingQType);

        this.we =  w.load("embeddings.word_embeddings.weight");
        this.wte = w.load("embeddings.token_type_embeddings.weight");
        this.wpe = w.load("embeddings.position_embeddings.weight");

        this.inputLayerNorm = new LayerNorm(this, w.load("embeddings.LayerNorm.bias"), w.load("embeddings.LayerNorm.weight"));

        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        for (int i = 0; i < c.numberOfLayers; i++) {
            String b = "encoder.layer." + i + ".";
            String prefix = b + "attention.";

            AbstractTensor keyBias = w.load(prefix + "self.key.bias");
            AbstractTensor keyWeight = w.load(prefix + "self.key.weight");

            AbstractTensor queryBias = w.load(prefix + "self.query.bias");
            AbstractTensor queryWeight = w.load(prefix + "self.query.weight");

            AbstractTensor valueBias = w.load(prefix + "self.value.bias");
            AbstractTensor valueWeight = w.load(prefix + "self.value.weight");

            AbstractTensor outputBias = w.load(prefix + "output.dense.bias");
            AbstractTensor outputWeight = w.load(prefix + "output.dense.weight");
            CausalSelfAttention attention = new CausalSelfAttention(this,
                    keyBias, queryBias, valueBias,
                    keyWeight, queryWeight, valueWeight,
                    outputBias, outputWeight, Optional.empty());

            prefix = b;
            MLPBlock mlpBlock = new MLPBlock(this, ActivationFunction.Type.GELU,
                    w.load(prefix + "intermediate.dense.bias"), w.load(prefix + "intermediate.dense.weight"),
                    w.load(prefix + "output.dense.bias"), w.load( prefix + "output.dense.weight")
            );

            LayerNorm postAttentionNorm = new LayerNorm(this, w.load(b + "attention.output.LayerNorm.bias"), w.load(b + "attention.output.LayerNorm.weight"));
            LayerNorm postMlpNorm = new LayerNorm(this, w.load(b + "output.LayerNorm.bias"), w.load(b + "output.LayerNorm.weight"));

            transformerBlocks[i] = new TransformerBlock(this, attention, postAttentionNorm, mlpBlock, postMlpNorm);
        }
    }

    @Override
    protected AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
        AbstractTensor embedding = makeTensor(c.embeddingLength);

        for (int i = 0; i < c.embeddingLength; i++) {
            float v = we.get(inputToken, i) + wte.get(0, i) + wpe.get(position, i);
            embedding.set(v, i);
        }

        AbstractTensor lnemb = this.inputLayerNorm.forward(embedding);
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
    protected AbstractTensor getOutputLogitsWeights() {
        throw new IllegalStateException();
    }

    public float[] embed(String input) {
        long[] encoded = tokenizer.encode(input);
        Preconditions.checkArgument(encoded.length < c.contextLength);

        AbstractTensor kvmem = makeTensor(c.numberOfLayers, encoded.length, 2, c.embeddingLength); // 2 for key and value

        int promptLength = encoded.length;
        float avgp = 1.0f/promptLength;

        float[] outputEmbedding = new float[c.embeddingLength];

        for (int i = 0; i < promptLength; i++) {
            int next = (int)encoded[i];
            AbstractTensor output = forward(next, i, kvmem);

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
