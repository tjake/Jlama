package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import com.google.common.base.Preconditions;

import java.util.Optional;

public class CausalSelfAttention {

    private static final boolean USE_FLASH_ATTN = true;

    private final AbstractModel m;
    private final Config c;
    private final AbstractTensor queryAttnBias;
    private final AbstractTensor keyAttnBias;

    private final AbstractTensor valueAttnBias;
    private final AbstractTensor outputProjectionBias;

    private final AbstractTensor queryAttnWeights;
    private final AbstractTensor keyAttnWeights;

    private final AbstractTensor valueAttnWeights;

    private final AbstractTensor outputProjectionWeights;

    private final Optional<float[][]> ropeFrequencies;

    private final int headSize;
    private final float attentionScale;


    public CausalSelfAttention(AbstractModel m, AbstractTensor queryAttnBias, AbstractTensor keyAttnBias, AbstractTensor valueAttnBias,
                               AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights,
                               AbstractTensor outputProjectionBias, AbstractTensor outputProjectionWeights, Optional<float[][]> ropeFrequencies)
    {
        this.m = m;
        this.c = m.c;
        this.queryAttnBias = queryAttnBias;
        this.keyAttnBias = keyAttnBias;
        this.valueAttnBias = valueAttnBias;
        this.queryAttnWeights = queryAttnWeights;
        this.keyAttnWeights = keyAttnWeights;
        this.valueAttnWeights = valueAttnWeights;

        this.outputProjectionBias = outputProjectionBias;
        this.outputProjectionWeights = outputProjectionWeights;

        this.headSize = c.embeddingLength / c.numberOfHeads;
        this.attentionScale = (float) (1.0 / StrictMath.sqrt(headSize));

        this.ropeFrequencies = ropeFrequencies;
    }

    public AbstractTensor forward(AbstractTensor input, int position, AbstractTensor kvMem) {
        Preconditions.checkArgument(input.dims() == 1 && input.shape()[0] == c.embeddingLength);

        try (AbstractTensor flashAttn_m = m.makeTensor(c.numberOfHeads);
             AbstractTensor flashAttn_l = m.makeTensor(c.numberOfHeads);
             AbstractTensor query = m.makeTensor(c.embeddingLength);
             AbstractTensor value = m.makeTensor(c.embeddingLength)) {

            //This is our memory of the key and value vectors for each position
            AbstractTensor kv = kvMem.slice(position);

            // compute the query vector
            VectorMath.pfor(0, c.embeddingLength, i -> {
                float v = queryAttnBias.get(i) + TensorOperationsProvider.get().dotProduct(input, queryAttnWeights.slice(i), c.embeddingLength);
                query.set(v, i);
            });

            // compute the key and value vectors
            VectorMath.pfor(0, c.embeddingLength, i -> {
                float v = keyAttnBias.get(i) + TensorOperationsProvider.get().dotProduct(input, keyAttnWeights.slice(i), c.embeddingLength);
                kv.set(v, i);
            });

            VectorMath.pfor(0, c.embeddingLength, i -> {
                float v = valueAttnBias.get(i) + TensorOperationsProvider.get().dotProduct(input, valueAttnWeights.slice(i), c.embeddingLength);
                kv.set(v, i + c.embeddingLength);
            });

            // apply RoPE if present (accounting for huggingface permutation)
            // https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/models/llama/convert_llama_weights_to_hf.py#L114
            ropeFrequencies.ifPresent(rf -> {
                int headPiece = headSize / 2;
                int poffset = position * headPiece;
                // apply RoPE rotation to the q and k vectors for each head
                for (int h = 0; h < c.numberOfHeads; h++) {
                    // get the q and k vectors for this head
                    int offset = h * headSize;
                    // rotate q and k by the freq theta and freq r
                    for (int i = offset; i < (offset + headPiece); i++) {
                        float q0 = query.get(i);
                        float q1 = query.get(i + headPiece);  //hf permutation is 0,64,1,65 etc...
                        float k0 = kv.get(i);
                        float k1 = kv.get(i + headPiece);
                        float[] f = rf[poffset + i];
                        float fcr = f[0];
                        float fci = f[1];
                        query.set(q0 * fcr - q1 * fci, i);
                        query.set(q0 * fci + q1 * fcr, i + headPiece);
                        kv.set(k0 * fcr - k1 * fci, i);
                        kv.set(k0 * fci + k1 * fcr, i + headPiece);
                    }
                }
            });

            if (USE_FLASH_ATTN) {
                // with all key-value entries populated, compute attention
                // the softmax is incrementally aggregated using the flash attention technique
                AbstractTensor k0 = kvMem.slice(0);

                // value is initially the first value for all heads
                value.copyFrom(k0, c.embeddingLength, 0, c.embeddingLength);

                //POSITION ZERO
                for (int i = 0; i < c.numberOfHeads; i++) {
                    float a = TensorOperationsProvider.get().dotProduct(query, k0, i * headSize, i * headSize, headSize) * attentionScale;
                    flashAttn_m.set(a, i);
                    flashAttn_l.set(1, i);
                }

                //POSITION > 0
                //This is where the context length gets expensive! We need to run this query token by all prior tokens.
                float[][] flashAttnHeads = new float[position][c.numberOfHeads];
                VectorMath.pfor(0, position, i -> {
                    AbstractTensor kk = kvMem.slice(i + 1);
                    VectorMath.pfor(0, c.numberOfHeads, h -> {
                        //KEY OFFSET
                        flashAttnHeads[i][h] = TensorOperationsProvider.get().dotProduct(query, kk, h * headSize, h * headSize, headSize) * attentionScale;
                    });
                });

                //Now aggregate results per head
                for (int i = 0; i < position; i++) {
                    AbstractTensor kk = kvMem.slice(i + 1);
                    for (int h = 0; h < c.numberOfHeads; h++) {
                        float a = flashAttnHeads[i][h];
                        if (a > flashAttn_m.get(h)) {
                            //VALUE OFFSET (since cache is k + v)
                            float e = (float) Math.exp(flashAttn_m.get(h) - a);
                            TensorOperationsProvider.get().sxpby(e, kk, value, c.embeddingLength + (h * headSize), h * headSize, headSize);
                            flashAttn_l.set(1 + e * flashAttn_l.get(h), h);
                            flashAttn_m.set(a, h);
                        } else {
                            //VALUE OFFSET (since cache is k + v)
                            float e = (float) Math.exp(a - flashAttn_m.get(h));
                            TensorOperationsProvider.get().saxpy(e, kk, value, c.embeddingLength + (h * headSize), h * headSize, headSize);
                            flashAttn_l.set(flashAttn_l.get(h) + e, h);
                        }
                    }
                }


                // scale y by 1/l
                for (int h = 0; h < c.numberOfHeads; h++) {
                    float scale = 1.0f / flashAttn_l.get(h);
                    TensorOperationsProvider.get().scale(scale, value, (h * headSize), headSize);
                }
            } else {

                try (AbstractTensor attn = m.makeTensor(position + 1)) {

                    VectorMath.pfor(0, c.numberOfHeads, h -> {
                        int hOffset = h * headSize;
                        for (int t = 0; t <= position; t++)
                        {
                            AbstractTensor kk = kvMem.slice(t);

                            float score = TensorOperationsProvider.get().dotProduct(query, kk, hOffset, hOffset, headSize);
                            score *= attentionScale;

                            attn.set(score, t);
                        }

                        // softmax the scores to get attention weights, from 0..pos inclusively
                        VectorMath.softMax(attn);

                        for (int t = 0; t <= position; t++)
                        {
                            AbstractTensor kk = kvMem.slice(t);
                            TensorOperationsProvider.get().saxpy(attn.get(t), kk, value, c.embeddingLength + (h * headSize), h * headSize, headSize);
                        }
                    });
                }
            }

            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            AbstractTensor result = m.makeTensor(c.embeddingLength);
            VectorMath.pfor(0, c.embeddingLength, i -> {
                float v = outputProjectionBias.get(i) + TensorOperationsProvider.get().dotProduct(value, outputProjectionWeights.slice(i), c.embeddingLength);
                result.set(v, i);
            });

            return result;
        }
    }
}
