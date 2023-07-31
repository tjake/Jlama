package com.github.tjake.llmj.model.gpt2;

import com.github.tjake.llmj.math.ActivationFunction;
import com.github.tjake.llmj.math.VectorMath;
import com.github.tjake.llmj.safetensors.Config;
import com.github.tjake.llmj.safetensors.Tokenizer;
import com.github.tjake.llmj.safetensors.Weights;
import com.github.tjake.llmj.model.*;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class GPT2Model {

    public final Config c;

    private final Tokenizer tokenizer;

    public final Tensor wte;
    public final Tensor wpe;

    public final LayerNorm layerNorm;

    public final TransformerBlock[] transformerBlocks;

    public GPT2Model(Config c, Weights w, Tokenizer tokenizer) {
        this.c = c;
        this.tokenizer = tokenizer;

        this.wte = w.load("wte.weight");
        this.wpe = w.load("wpe.weight");

        this.layerNorm = new LayerNorm(c, w.load("ln_f.bias"), w.load("ln_f.weight"));
        this.transformerBlocks = new TransformerBlock[c.numberOfLayers];

        for (int i = 0; i < transformerBlocks.length; i++)
        {
            String b = "h." + i + ".";
            String prefix = b + "attn.";

            Tensor[] attnBias = w.load(prefix + "c_attn.bias").split(3, 0);
            Tensor[] attnWeights = w.load(prefix + "c_attn.weight").transpose().split(3, 0);
            CausalSelfAttention attention = new CausalSelfAttention(c, attnBias[0], attnBias[1], attnBias[2],
                    attnWeights[0], attnWeights[1], attnWeights[2],
                    w.load(prefix + "c_proj.bias"), w.load(prefix + "c_proj.weight").transpose(), Optional.empty());

            prefix = b + "mlp.";
            MLPBlock mlpBlock = new MLPBlock(c, ActivationFunction.Type.GELU,
                    w.load(prefix + "c_fc.bias"), w.load(prefix + "c_fc.weight").transpose(),
                    w.load(prefix + "c_proj.bias"), w.load( prefix + "c_proj.weight").transpose()
            );

            LayerNorm layerNorm1 = new LayerNorm(c, w.load(b + "ln_1.bias"), w.load(b + "ln_1.weight"));
            LayerNorm layerNorm2 = new LayerNorm(c, w.load(b + "ln_2.bias"), w.load(b + "ln_2.weight"));

            transformerBlocks[i] = new TransformerBlock(attention, layerNorm1, layerNorm2, mlpBlock);
        }
    }

    public Tensor forward(int token_id, int pos, Tensor kvbuf) {

        Tensor embedding = c.bufferCache.get(c.embeddingLength);

        for (int i = 0; i < c.embeddingLength; i++) {
            float v = wte.get(token_id, i) + wpe.get(pos, i);
            embedding.set(v, i);
        }

        for (int i = 0; i < c.numberOfLayers; i++) {
            Tensor kvlayer = kvbuf.slice(i);
            Tensor ref = embedding; //reference so we can free
            embedding = transformerBlocks[i].forward(embedding, pos, kvlayer);
            ref.close();
        }
        return embedding;
    }

    int sample(Tensor output, float temperature, float uniformSample, Tensor logits) {
        try(Tensor embedding = layerNorm.forward(output)) {
            float max = Float.NEGATIVE_INFINITY;
            int ntokens = logits.shape()[0];
            for (int i = 0; i < ntokens; i++) {
                float v = VectorMath.dotProduct(embedding, wte.slice(i), c.embeddingLength);
                logits.set(v, i);
                if (v > max)
                    max = v;
            }

            float sum = 0;
            for (int i = 0; i < ntokens; i++) {
                float v = (float) Math.exp((logits.get(i) - max) / temperature);
                sum += v;
                logits.set(v, i);
            }

            float acc = 0;
            for (int i = 0; i < ntokens; i++) {
                float v = logits.get(i) / sum;
                logits.set(v, i);
                acc += v;
                if (acc >= uniformSample)
                    return i;
            }

            throw new RuntimeException("Sampling Error? " + uniformSample + " " + acc);
        }
    }

    public void run(String prompt, float temperature, int ntokens, Consumer<String> onToken) {
        boolean init = VectorMath.hasVectorAPI;
        long[] encoded = tokenizer.encode(prompt);
        Preconditions.checkArgument(encoded.length < c.contextLength);

        Tensor kvmem = new FloatBufferTensor(c.numberOfLayers, c.contextLength, c.embeddingLength * 2); //k and v are concatenated

        // forbid generation of <|endoftext|> by cutting it out of the logit buffer (it's the last token)
        Tensor logits = new FloatBufferTensor(c.vocabularySize - 1);

        int[] tokens = new int[c.contextLength];

        // always start with <|endoftext|>
        tokens[0] = 50256; // <|endoftext|>

        for (int i = 0; i < encoded.length; i++)
            tokens[i + 1] = Ints.checkedCast(encoded[i]);

        int promptLength = encoded.length;
        System.out.println(prompt);
        long start = System.currentTimeMillis();
        int tokensGenerated = 0;
        for (int i = 0; i < ntokens - 1; i++) {
            //System.out.println("Token = " + tokens[i]);
            try(Tensor output = forward(tokens[i], i, kvmem)) {
                tokensGenerated++;
                if (i < promptLength)
                    continue;

                int sampled_token = sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);
                tokens[i + 1] = sampled_token;
                String p = tokenizer.decode(sampled_token);
                onToken.accept(p);
                //long end = System.currentTimeMillis();
                //System.out.printf("elapsed: %ds, %fms per token\n", TimeUnit.MILLISECONDS.toSeconds(end - start), ((end - start) / (float) i));
            }
        }

        long end = System.currentTimeMillis();
        System.out.printf("\n\nelapsed: %ds, %fms per token\n", TimeUnit.MILLISECONDS.toSeconds(end - start), ((end - start) / (float) tokensGenerated));
    }

}
