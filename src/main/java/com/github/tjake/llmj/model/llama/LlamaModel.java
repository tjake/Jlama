package com.github.tjake.llmj.model.llama;

import com.github.tjake.llmj.math.ActivationFunction;
import com.github.tjake.llmj.math.VectorMath;
import com.github.tjake.llmj.safetensors.Config;
import com.github.tjake.llmj.safetensors.SafeTensorIndex;
import com.github.tjake.llmj.model.*;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(LlamaModel.class);
    private final Config c;
    private final LlamaTokenizer tokenizer;

    private final Tensor wte;

    private final LayerNorm outputLayerNorm;

    private final Tensor noBias;

    private final TransformerBlock[] transformerBlocks;

    private final Tensor classificationWeights;

    public LlamaModel(Config config, SafeTensorIndex weights, LlamaTokenizer tokenizer) {
        this.c = config;
        this.tokenizer = tokenizer;

        logger.info("Loading model...");

        //LLama doesn't use bias, will optimize this away later
        this.noBias = new FloatBufferTensor(c.hiddenLength);

        this.wte = weights.load("model.embed_tokens.weight");
        this.outputLayerNorm = new RMSNorm(c, noBias, weights.load("model.norm.weight"));
        this.classificationWeights = weights.load("lm_head.weight");

        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        float[][] ropeFreqs = VectorMath.precomputeFreqsCis(c.embeddingLength / c.numberOfHeads, c.contextLength * 2, 10000.0 );

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
                    noBias, weights.load(prefix + "gate_proj.weight"),
                    noBias, weights.load(prefix + "down_proj.weight"),
                    weights.load(prefix + "up_proj.weight"));

            this.transformerBlocks[i] = new TransformerBlock(attention,
                    new RMSNorm(c, noBias, weights.load(base + "input_layernorm.weight")),
                    new RMSNorm(c, noBias, weights.load(base + "post_attention_layernorm.weight")),
                    mlp);
        }
        logger.info("Model loaded!");
    }

    public Tensor forward(int token_id, int pos, Tensor kvbuf) {
        Tensor embedding = new FloatBufferTensor(c.embeddingLength);

        for (int i = 0; i < c.embeddingLength; i++) {
            float v = wte.get(token_id, i);
            embedding.set(v, i);
        }

        for (int i = 0; i < c.numberOfLayers; i++) {
            Tensor kvlayer = kvbuf.slice(i);
            embedding = transformerBlocks[i].forward(embedding, pos, kvlayer);
        }
        return embedding;
    }

    int sample(Tensor embedding, float temperature, float uniformSample, Tensor logits) {
        embedding = outputLayerNorm.forward(embedding);
        float maxv = Float.NEGATIVE_INFINITY;
        int maxi = Integer.MIN_VALUE;

        //This is a mix of argmax and sampling with softmax
        for (int i = 0; i < c.vocabularySize; i++)
        {
            float v = VectorMath.dotProduct(embedding, classificationWeights.slice(i), c.embeddingLength);
            logits.set(v, i);
            if (v > maxv) {
                maxv = v;
                maxi = i;
            }
        }

        if (temperature == 0.0) {
            return maxi;
        }

        float sum = 0;
        for (int i = 0; i < c.vocabularySize; i++) {
            float v = (float)Math.exp( (logits.get(i) - maxv) / temperature );
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

    public void run(String prompt, float temperature, int ntokens, boolean useEOS, Consumer<String> onToken) {
        long[] encoded = tokenizer.encode(prompt);
        Preconditions.checkArgument(encoded.length < c.contextLength);

        if (ntokens > c.contextLength)
            ntokens = c.contextLength;

        Tensor kvmem = new FloatBufferTensor(c.numberOfLayers, ntokens, c.embeddingLength * 2); //k and v are concatenated
        Tensor logits = new FloatBufferTensor(c.vocabularySize);

        int[] tokens = new int[c.contextLength];

        tokens[0] = 1; // Add BOS

        for (int i = 0; i < encoded.length; i++)
            tokens[i + 1] = Ints.checkedCast(encoded[i]);

        int promptLength = encoded.length + 1;

        if (useEOS) {
            tokens[encoded.length + 1] = 2; //Add EOS
            promptLength++;
        }

        long start = System.currentTimeMillis();
        StringBuilder b = new StringBuilder();
        int tokensGenerated = 0;
        for (int i = 0; i < ntokens - 1; i++) {
            //logger.info("Generating token " + tokens[i]);

            Tensor output = forward(tokens[i], i, kvmem);

            int sampled_token = sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);
            tokensGenerated++;

            String p = tokenizer.decode(sampled_token);

            // Keep prompt tokens till they are gone
            if (i < promptLength) {
                p = tokenizer.decode(tokens[i]);
            } else if (sampled_token == 2) {
                break;
            }

            if (tokens[i + 1] == 0) {
                tokens[i + 1] = sampled_token;
            }

            //b.append(p);
            onToken.accept(p);
            //onToken.accept( "" + i +": '" + b + "' [" + tokens[i] + "] => '" + p + "' [" + sampled_token + "] next=" + tokenizer.decode(tokens[i+1]) + "[" + tokens[i + 1] + "] took " + ((System.currentTimeMillis() - start)/(i+1)) + "ms");
        }

        long end = System.currentTimeMillis();
        System.out.printf("\n\nelapsed: %ds, %fms per token\n", TimeUnit.MILLISECONDS.toSeconds(end - start), ((end - start) / (float)tokensGenerated));
    }
}
