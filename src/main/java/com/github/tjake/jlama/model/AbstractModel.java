package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.Tokenizer;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import com.google.common.util.concurrent.AtomicDouble;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public abstract class AbstractModel {
    private static final Logger logger = LoggerFactory.getLogger(AbstractModel.class);
    protected final Config c;
    protected final WeightLoader weights;
    protected final Tokenizer tokenizer;
    protected final DType modelDType;

    protected AbstractModel(Config c, WeightLoader w, Tokenizer t)
    {
        this.c = c;
        this.weights = w;
        this.tokenizer = t;
        this.modelDType = w.getModelDType();
    }

    protected abstract AbstractTensor inputTokenToEmbedding(int inputToken, int position);

    protected abstract TransformerBlock[] getTransformerBlocks();

    protected abstract LayerNorm getOutputLayerNorm();

    protected abstract AbstractTensor getOutputLogitsWeights();

    protected AbstractTensor makeTensor(int ...shape) {
        return c.tensorCache.get(modelDType, shape);
    }

    protected AbstractTensor forward(int token_id, int pos, AbstractTensor kvbuf) {
        AbstractTensor embedding = inputTokenToEmbedding(token_id, pos);
        TransformerBlock[] transformerBlocks = getTransformerBlocks();

        for (int i = 0; i < c.numberOfLayers; i++) {
            AbstractTensor kvlayer = kvbuf.slice(i);
            AbstractTensor ref = embedding; //reference so we can free
            embedding = transformerBlocks[i].forward(embedding, pos, kvlayer);
            ref.close();
        }
        return embedding;
    }

    protected int sample(AbstractTensor output, float temperature, float uniformSample, AbstractTensor logits) {
        try(AbstractTensor embedding = getOutputLayerNorm().forward(output)) {

            AtomicDouble maxv = new AtomicDouble(Double.NEGATIVE_INFINITY);
            AtomicInteger maxi = new AtomicInteger(Integer.MIN_VALUE);

            //This is a mix of argmax and sampling with softmax
            VectorMath.pfor(0, c.vocabularySize, i -> {
                float v = VectorMath.dotProduct(embedding, getOutputLogitsWeights().slice(i), c.embeddingLength);
                logits.set(v, i);
                maxv.getAndUpdate(x -> {
                    if (v > x) {
                        maxi.set(i);
                        return v;
                    }
                    return x;
                });
            });

            if (temperature == 0.0) {
                return maxi.get();
            }

            float sum = 0;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = (float) Math.exp((logits.get(i) - maxv.get()) / temperature);
                sum += v;
                logits.set(v, i);
            }

            float acc = 0;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = logits.get(i) / sum;
                acc += v;
                if (acc >= uniformSample)
                    return i;
            }

            return c.vocabularySize - 1;
        }
    }

    public void generate(String prompt, float temperature, int ntokens, boolean useEOS, BiConsumer<String, Float> onTokenWithTimings) {
        long[] encoded = tokenizer.encode(prompt);
        Preconditions.checkArgument(encoded.length < c.contextLength);

        if (ntokens > c.contextLength)
            ntokens = c.contextLength;

        AbstractTensor kvmem = makeTensor(c.numberOfLayers, ntokens, c.embeddingLength * 2); //k and v are concatenated
        AbstractTensor logits = makeTensor(c.vocabularySize);

        int[] tokens = new int[c.contextLength];

        for (int i = 0; i < encoded.length; i++)
            tokens[i] = Ints.checkedCast(encoded[i]);

        int promptLength = encoded.length;

        if (useEOS) {
            tokens[encoded.length + 1] = c.eosToken; //Add EOS
            promptLength++;
        }

        long start = System.currentTimeMillis();
        int tokensGenerated = 0;
        int next = c.bosToken;

        for (int i = 0; i < ntokens; i++) {
            AbstractTensor output = forward(next, i, kvmem);
            tokensGenerated++;
            // Keep prompt tokens till they are gone
            if (i < promptLength) {
                next = tokens[i];
            } else {
                next = sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);

                if (logger.isTraceEnabled())
                    logger.trace("Sampled token {} with temperature {}", next, temperature);

                //Model may tell us it's done
                if (next == c.eosToken)
                    break;
            }
            try {
                String c = tokenizer.decode(next);
                onTokenWithTimings.accept(c, (System.currentTimeMillis() - start)/(float)(i+1));
            } catch (Exception e) {
                logger.error("Failed to decode token {}", next, e);
            }
        }

        long end = System.currentTimeMillis();
        System.out.printf("\n\nelapsed: %ds, %fms per token\n", TimeUnit.MILLISECONDS.toSeconds(end - start), ((end - start) / (float)tokensGenerated));
    }
}
