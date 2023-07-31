package com.github.tjake.llmj.model;

import com.github.tjake.llmj.math.VectorMath;
import com.github.tjake.llmj.safetensors.Config;
import com.google.common.base.Preconditions;

import java.util.Optional;
import java.util.stream.IntStream;

public class CausalSelfAttention {

    private final Config c;
    private final Tensor queryAttnBias;
    private final Tensor keyAttnBias;

    private final Tensor valueAttnBias;
    private final Tensor outputProjectionBias;

    private final Tensor queryAttnWeights;
    private final Tensor keyAttnWeights;

    private final Tensor valueAttnWeights;

    private final Tensor outputProjectionWeights;

    private final Optional<float[][]> ropeFrequencies;

    public CausalSelfAttention(Config c, Tensor queryAttnBias, Tensor keyAttnBias, Tensor valueAttnBias,
                               Tensor queryAttnWeights, Tensor keyAttnWeights, Tensor valueAttnWeights,
                               Tensor outputProjectionBias, Tensor outputProjectionWeights, Optional<float[][]> ropeFrequencies)
    {
        this.c = c;
        this.queryAttnBias = queryAttnBias;
        this.keyAttnBias = keyAttnBias;
        this.valueAttnBias = valueAttnBias;
        this.queryAttnWeights = queryAttnWeights;
        this.keyAttnWeights = keyAttnWeights;
        this.valueAttnWeights = valueAttnWeights;

        this.outputProjectionBias = outputProjectionBias;
        this.outputProjectionWeights = outputProjectionWeights;

        this.ropeFrequencies = ropeFrequencies;
    }

    public Tensor forward(Tensor input, int position, Tensor kvMem) {
        Preconditions.checkArgument(input.dims() == 1 && input.shape()[0] == c.embeddingLength);

        try (Tensor flashAttn_m = c.bufferCache.get(c.numberOfHeads);
            Tensor flashAttn_l = c.bufferCache.get(c.numberOfHeads);
            Tensor query = c.bufferCache.get(c.embeddingLength);
            Tensor value = c.bufferCache.get(c.embeddingLength)) {

            int headSize = c.embeddingLength / c.numberOfHeads;
            float attentionScale = 1.0f / (float) Math.sqrt(headSize);

            //This is our memory of the key and value vectors for each position
            Tensor kv = kvMem.slice(position);

            // compute the query vector
            for (int i = 0; i < c.embeddingLength; i++) {
                float v = queryAttnBias.get(i) + VectorMath.dotProduct(input, queryAttnWeights.slice(i), c.embeddingLength);
                query.set(v, i);
            }


            // compute the key and value vectors
            for (int i = 0; i < c.embeddingLength; i++) {
                float v = keyAttnBias.get(i) + VectorMath.dotProduct(input, keyAttnWeights.slice(i), c.embeddingLength);
                kv.set(v, i);
            }

            for (int i = 0; i < c.embeddingLength; i++) {
                float v = valueAttnBias.get(i) + VectorMath.dotProduct(input, valueAttnWeights.slice(i), c.embeddingLength);
                kv.set(v, i + c.embeddingLength);
            }

            // apply RoPE if present
            ropeFrequencies.ifPresent(rf -> {
                int poffset = position * headSize / 2;
                // apply RoPE rotation to the q and k vectors for each head
                for (int h = 0; h < c.numberOfHeads; h++) {
                    // get the q and k vectors for this head
                    int offset = h * headSize;
                    // rotate q and k by the freq theta and freq r
                    for (int i = offset; i < (offset + headSize); i += 2) {
                        float q0 = query.get(i);
                        float q1 = query.get(i + 1);
                        float k0 = kv.get(i);
                        float k1 = kv.get(i + 1);
                        float[] f = rf[poffset + i / 2];
                        float fcr = f[0];
                        float fci = f[1];
                        query.set(q0 * fcr - q1 * fci, i);
                        query.set(q0 * fci + q1 * fcr, i + 1);
                        kv.set(k0 * fcr - k1 * fci, i);
                        kv.set(k0 * fci + k1 * fcr, i + 1);
                    }
                }
            });

            // with all key-value entries populated, compute attention
            // the softmax is incrementally aggregated using the flash attention technique
            Tensor k0 = kvMem.slice(0);

            // value is initially the first value for all heads
            System.arraycopy(k0.getFloatArray(), k0.getArrayOffset() + c.embeddingLength, value.getFloatArray(), value.getArrayOffset(), c.embeddingLength);

            //POSITION ZERO
            for (int i = 0; i < c.numberOfHeads; i++) {
                float a = VectorMath.dotProduct(query, k0, i * headSize, i * headSize, headSize) * attentionScale;
                flashAttn_m.set(a, i);
                flashAttn_l.set(1, i);
            }

            //POSITION > 0
            //This is where the context length gets expensive! We need to run this query token by all prior tokens.
            float[][] flashAttnHeads = new float[position][c.numberOfHeads];
            IntStream.range(0, position).parallel().forEach(i -> {
                Tensor kk = kvMem.slice(i+1);
                IntStream.range(0, c.numberOfHeads).parallel().forEach(h -> {
                    //KEY OFFSET
                    flashAttnHeads[i][h] = VectorMath.dotProduct(query, kk, h * headSize, h * headSize, headSize) * attentionScale;
                });
            });

            //Now aggregate results per head
            for(int i = 0; i < position; i++) {
                Tensor kk = kvMem.slice(i+1);
                for (int h = 0; h < c.numberOfHeads; h++) {
                    float a = flashAttnHeads[i][h];
                    if (a > flashAttn_m.get(h)) {
                        //VALUE OFFSET
                        float e = (float) Math.exp(flashAttn_m.get(h) - a);
                        VectorMath.sxpby(e, kk, value, c.embeddingLength + (h * headSize), h * headSize, headSize);
                        flashAttn_l.set(1 + e * flashAttn_l.get(h), h);
                        flashAttn_m.set(a, h);
                    } else {
                        //VALUE OFFSET
                        float e = (float) Math.exp(a - flashAttn_m.get(h));
                        VectorMath.saxpy(e, kk, value, c.embeddingLength + (h * headSize), h * headSize, headSize);
                        flashAttn_l.set(flashAttn_l.get(h) + e, h);
                    }
                }
            }


            // scale y by 1/l
            for (int h = 0; h < c.numberOfHeads; h++) {
                float scale = 1.0f / flashAttn_l.get(h);
                VectorMath.scale(value.getFloatArray(), value.getArrayOffset() + (h * headSize), headSize, scale);
            }

            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            Tensor result = c.bufferCache.get(c.embeddingLength);
            for (int i = 0; i < c.embeddingLength; i++) {
                float v = outputProjectionBias.get(i) + VectorMath.dotProduct(value, outputProjectionWeights.slice(i), c.embeddingLength);
                result.set(v, i);
            }

            return result;
        }
    }
}
