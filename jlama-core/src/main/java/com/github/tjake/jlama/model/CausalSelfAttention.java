package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import com.google.common.base.Preconditions;

import java.util.Arrays;
import java.util.Optional;

public class CausalSelfAttention {
    private final AbstractModel m;
    private final Config c;
    private final Optional<AbstractTensor> queryAttnBias;
    private final Optional<AbstractTensor> keyAttnBias;

    private final Optional<AbstractTensor> valueAttnBias;
    private final Optional<AbstractTensor> outputProjectionBias;

    private final AbstractTensor queryAttnWeights;
    private final AbstractTensor keyAttnWeights;

    private final AbstractTensor valueAttnWeights;

    private final AbstractTensor outputProjectionWeights;

    private final Optional<float[][]> ropeFrequencies;

    private final int headSize;
    private final float attentionScale;

    private final float[][] flashAttnHeads;

    public CausalSelfAttention(AbstractModel m, AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights,
                               AbstractTensor outputProjectionWeights, Optional<float[][]> ropeFrequencies)
    {
        this(m, Optional.empty(), Optional.empty(), Optional.empty(), queryAttnWeights, keyAttnWeights, valueAttnWeights, Optional.empty(), outputProjectionWeights, ropeFrequencies);
    }

    public CausalSelfAttention(AbstractModel m, AbstractTensor queryAttnBias, AbstractTensor keyAttnBias, AbstractTensor valueAttnBias,
                               AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights,
                               AbstractTensor outputProjectionBias, AbstractTensor outputProjectionWeights, Optional<float[][]> ropeFrequencies) {
        this(m, Optional.of(queryAttnBias), Optional.of(keyAttnBias), Optional.of(valueAttnBias), queryAttnWeights, keyAttnWeights, valueAttnWeights, Optional.of(outputProjectionBias), outputProjectionWeights, ropeFrequencies);
    }


    public CausalSelfAttention(AbstractModel m, Optional<AbstractTensor> queryAttnBias, Optional<AbstractTensor> keyAttnBias, Optional<AbstractTensor> valueAttnBias,
                               AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights,
                               Optional<AbstractTensor> outputProjectionBias, AbstractTensor outputProjectionWeights, Optional<float[][]> ropeFrequencies)
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
        this.flashAttnHeads = new float[c.contextLength][c.numberOfHeads];
    }

    public AbstractTensor forward(AbstractTensor input, int position, AbstractTensor kvMem) {
        Preconditions.checkArgument(input.dims() == 1 && input.shape()[0] == c.embeddingLength);

        try (AbstractTensor flashAttn_m = m.makeTensor(c.numberOfHeads);
             AbstractTensor flashAttn_l = m.makeTensor(c.numberOfHeads);
             AbstractTensor query = m.makeTensor(c.embeddingLength);
             AbstractTensor value = m.makeTensor(c.embeddingLength))
        {

            //This is our memory of the key and value vectors for each position
            AbstractTensor kvp = kvMem.slice(true, position);

            AbstractTensor key = kvp.slice(0);
            AbstractTensor val = kvp.slice(1);

            // compute the query vector
            VectorMath.pchunk(c.embeddingLength, (chunkStart, chunkLength) -> {
                TensorOperationsProvider.get().dotProductChunk(query, input, queryAttnWeights, c.embeddingLength, chunkStart, chunkLength);
                TensorOperationsProvider.get().dotProductChunk(key, input, keyAttnWeights, c.embeddingLength, chunkStart, chunkLength);
                TensorOperationsProvider.get().dotProductChunk(val, input, valueAttnWeights, c.embeddingLength, chunkStart, chunkLength);
            });

            queryAttnBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(query, bias));
            keyAttnBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(key, bias));
            valueAttnBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(val, bias));

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
                        float k0 = key.get(i);
                        float k1 = key.get(i + headPiece);
                        float[] f = rf[poffset + i];
                        float fcr = f[0];
                        float fci = f[1];
                        query.set(q0 * fcr - q1 * fci, i);
                        query.set(q0 * fci + q1 * fcr, i + headPiece);
                        key.set(k0 * fcr - k1 * fci, i);
                        key.set(k0 * fci + k1 * fcr, i + headPiece);
                    }
                }
            });

            // with all key-value entries populated, compute attention
            // the softmax is incrementally aggregated using the flash attention technique
            AbstractTensor k0 = kvMem.slice(true, 0).slice(0);
            AbstractTensor v0 = kvMem.slice(true,0).slice(1);

            // value is initially the first value for all heads
            value.copyFrom(v0, 0, 0, c.embeddingLength);

            //POSITION ZERO
            for (int i = 0; i < c.numberOfHeads; i++) {
                float a = TensorOperationsProvider.get().dotProduct(query, k0, i * headSize, i * headSize, headSize) * attentionScale;
                flashAttn_m.set(a, i);
                flashAttn_l.set(1, i);
            }

            //POSITION > 0
            //This is where the context length gets expensive! We need to run this query token by all prior tokens.
            VectorMath.pfor(0, position, i -> {
                //KEY OFFSET
                AbstractTensor kk = kvMem.slice(true, i + 1).slice(0);
                for(int h = 0; h < c.numberOfHeads; h++){
                    flashAttnHeads[i][h] = TensorOperationsProvider.get().dotProduct(query, kk, h * headSize, h * headSize, headSize) * attentionScale;
                }
            });

            //Now aggregate results per head
            for (int i = 0; i < position; i++) {
                //VALUE OFFSET
                AbstractTensor vv = kvMem.slice(true, i + 1).slice(1);
                for (int h = 0; h < c.numberOfHeads; h++) {
                    float a = flashAttnHeads[i][h];
                    if (a > flashAttn_m.get(h)) {
                        float e = (float) Math.exp(flashAttn_m.get(h) - a);
                        TensorOperationsProvider.get().sxpby(e, vv, value, (h * headSize), h * headSize, headSize);
                        flashAttn_l.set(1 + e * flashAttn_l.get(h), h);
                        flashAttn_m.set(a, h);
                    } else {
                        float e = (float) Math.exp(a - flashAttn_m.get(h));
                        TensorOperationsProvider.get().saxpy(e, vv, value, (h * headSize), h * headSize, headSize);
                        flashAttn_l.set(flashAttn_l.get(h) + e, h);
                    }
                }
            }

            // scale y by 1/l
            for (int h = 0; h < c.numberOfHeads; h++) {
                float scale = 1.0f / flashAttn_l.get(h);
                TensorOperationsProvider.get().scale(scale, value, (h * headSize), headSize);
            }

            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            AbstractTensor result = m.makeTensor(c.embeddingLength);
            try(AbstractTensor vq = m.maybeQuantize(value)) {
                VectorMath.pchunk( c.embeddingLength, (chunkStart, chunkSize) -> {
                    TensorOperationsProvider.get().dotProductChunk(result, vq, outputProjectionWeights, c.embeddingLength, chunkStart, chunkSize);
                });

                outputProjectionBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(result, bias));
            }

            return result;
        }
    }
}
