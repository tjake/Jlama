package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import com.google.common.base.Preconditions;

import java.util.*;
import java.util.function.Consumer;

public class CausalSelfAttention {
    private final AbstractModel m;
    private final Config c;
    private final Optional<AbstractTensor> queryAttnBias;
    private final Optional<AbstractTensor> keyAttnBias;

    private final Optional<AbstractTensor> valueAttnBias;
    private final Optional<AbstractTensor> outputProjectionBias;

    final AbstractTensor queryAttnWeights;
    final AbstractTensor keyAttnWeights;

    final AbstractTensor valueAttnWeights;

    private final AbstractTensor outputProjectionWeights;


    private final float attentionScale;

    private final float[][] flashAttnHeads;

    private final AbstractTensor[] qkvResults;
    private final AbstractTensor[] qkvWeights;


    public CausalSelfAttention(AbstractModel m, AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights,
                               AbstractTensor outputProjectionWeights)
    {
        this(m, Optional.empty(), Optional.empty(), Optional.empty(), queryAttnWeights, keyAttnWeights, valueAttnWeights, Optional.empty(), outputProjectionWeights);
    }

    public CausalSelfAttention(AbstractModel m, AbstractTensor queryAttnBias, AbstractTensor keyAttnBias, AbstractTensor valueAttnBias,
                               AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights,
                               AbstractTensor outputProjectionBias, AbstractTensor outputProjectionWeights) {
        this(m, Optional.of(queryAttnBias), Optional.of(keyAttnBias), Optional.of(valueAttnBias), queryAttnWeights, keyAttnWeights, valueAttnWeights, Optional.of(outputProjectionBias), outputProjectionWeights);
    }


    public CausalSelfAttention(AbstractModel m, Optional<AbstractTensor> queryAttnBias, Optional<AbstractTensor> keyAttnBias, Optional<AbstractTensor> valueAttnBias,
                               AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights,
                               Optional<AbstractTensor> outputProjectionBias, AbstractTensor outputProjectionWeights)
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

        this.attentionScale = (float) (1.0 / StrictMath.sqrt(c.headSize));

        this.flashAttnHeads = new float[c.contextLength][c.numberOfHeads];

        this.qkvResults = new AbstractTensor[3];
        this.qkvWeights = new AbstractTensor[]{queryAttnWeights, keyAttnWeights, valueAttnWeights};
    }

    public AbstractTensor forward(AbstractTensor input, int position, AbstractTensor kvMem, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        Preconditions.checkArgument(input.dims() == 1 && input.shape().first() == c.embeddingLength);

        try (AbstractTensor flashAttn_m = m.makeTensor(c.numberOfHeads);
             AbstractTensor flashAttn_l = m.makeTensor(c.numberOfHeads);
             AbstractTensor query = m.makeFullTensor(c.embeddingLength);
             AbstractTensor tmpKey = m.makeFullTensor(c.kvLength);
             AbstractTensor tmpVal = m.makeFullTensor(c.kvLength);
             AbstractTensor value = m.makeFullTensor(c.embeddingLength))
        {
            //This is our memory of the key and value vectors for each position
            AbstractTensor kvp = kvMem.slice(true, position);

            AbstractTensor key = kvp.slice(0);
            AbstractTensor val = kvp.slice(1);

            if (c.isGQA) {
                VectorMath.pchunk(0, c.embeddingLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get().dotProductChunk(query, input, queryAttnWeights, c.embeddingSegmentStart(), c.embeddingSegmentLength(), chunkStart, chunkLength);
                });
                VectorMath.pchunk(0, c.kvLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get().dotProductChunk(tmpKey, input, keyAttnWeights, c.embeddingSegmentStart(), c.embeddingSegmentLength(), chunkStart, chunkLength);
                    TensorOperationsProvider.get().dotProductChunk(tmpVal, input, valueAttnWeights, c.embeddingSegmentStart(), c.embeddingSegmentLength(), chunkStart, chunkLength);
                });
            } else {
                qkvResults[0] = query;
                qkvResults[1] = tmpKey;
                qkvResults[2] = tmpVal;

                // compute the query vector
                VectorMath.pchunk(0, c.embeddingLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get().dotProductBatchChunk(qkvResults, input, qkvWeights, c.embeddingSegmentStart(), c.embeddingSegmentLength(), chunkStart, chunkLength);
                });
            }
            // For distributed sum of tensor
            tensorReducer.ifPresent(func -> func.accept(List.of(query, tmpKey, tmpVal)));

            queryAttnBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(query, bias, c.embeddingSegmentStart(), c.embeddingSegmentLength()));
            keyAttnBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(tmpKey, bias, c.kvSegmentStart(), c.kvSegmentLength()));
            valueAttnBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(tmpVal, bias, c.kvSegmentStart(), c.kvSegmentLength()));

            key.copyFrom(tmpKey, tmpKey.getOffset(c.kvSegmentStart()), key.getOffset(c.kvSegmentStart()), c.kvSegmentLength());
            val.copyFrom(tmpVal, tmpVal.getOffset(c.kvSegmentStart()), val.getOffset(c.kvSegmentStart()), c.kvSegmentLength());

            // apply RoPE if present (accounting for huggingface permutation)
            // https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/models/llama/convert_llama_weights_to_hf.py#L114
            c.ropeFreqs.ifPresent(rf -> {
                int headPiece = c.headSize / 2;
                int poffset = position * headPiece;

                if (c.isGQA) {
                    // apply RoPE rotation to the q and k vectors for each head
                    for (int h = c.headStart(); h < c.headEnd(); h++) {
                        // get the q vectors for this head
                        int offset = h * c.headSize;
                        int goffset = c.maybeMapToGroupHead(h) * c.headSize;
                        // rotate q by the freq theta and freq r
                        for (int i = offset, g = goffset; i < (offset + headPiece); i++, g++) {
                            float q0 = query.get(i);
                            float q1 = query.get(i + headPiece);  //hf permutation is 0,64,1,65 etc...
                            float[] f = rf[poffset + g];
                            float fcr = f[0];
                            float fci = f[1];
                            query.set(q0 * fcr - q1 * fci, i);
                            query.set(q0 * fci + q1 * fcr, i + headPiece);
                        }
                    }

                    for (int h = c.groupHeadStart(); h < c.groupHeadEnd(); h++) {
                        // get the k vectors for this head
                        int offset = h * c.headSize;
                        // rotate k by the freq theta and freq r
                        for (int i = offset; i < (offset + headPiece); i++) {
                            float k0 = key.get(i);
                            float k1 = key.get(i + headPiece);  //hf permutation is 0,64,1,65 etc...
                            float[] f = rf[poffset + i];
                            float fcr = f[0];
                            float fci = f[1];
                            key.set(k0 * fcr - k1 * fci, i);
                            key.set(k0 * fci + k1 * fcr, i + headPiece);
                        }
                    }
                } else {
                    // apply RoPE rotation to the q and k vectors for each head
                    for (int h = c.headStart(); h < c.headEnd(); h++) {
                        // get the q and k vectors for this head
                        int offset = h * c.headSize;
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
                }
            });

            // with all key-value entries populated, compute attention
            // the softmax is incrementally aggregated using the flash attention technique
            AbstractTensor k0 = kvMem.slice(true,0).slice(0);
            AbstractTensor v0 = kvMem.slice(true,0).slice(1);

            // value is initially the position 0 value for all heads
            //POSITION ZERO
            for (int i = c.headStart(); i < c.headEnd(); i++) {
                value.copyFrom(v0, v0.getOffset(c.maybeMapToGroupHead(i) * c.headSize), value.getOffset(i * c.headSize), c.headSize);
                float a = TensorOperationsProvider.get().dotProduct(query, k0, i * c.headSize, c.maybeMapToGroupHead(i) * c.headSize, c.headSize) * attentionScale;
                flashAttn_m.set(a, i);
                flashAttn_l.set(1, i);
            }

            //POSITION > 0
            //This is where the context length gets expensive! We need to run this query token by all prior tokens.
            VectorMath.pfor(0, position, i -> {
                //KEY OFFSET
                AbstractTensor kk = kvMem.slice(true, i + 1).slice(0);
                for(int h = c.headStart(); h < c.headEnd(); h++){
                    flashAttnHeads[i][h] = TensorOperationsProvider.get().dotProduct(query, kk, h * c.headSize, c.maybeMapToGroupHead(h) * c.headSize, c.headSize) * attentionScale;
                }
            });

            //Now aggregate results per head
            for (int i = 0; i < position; i++) {
                //VALUE OFFSET
                AbstractTensor vv = kvMem.slice(true, i + 1).slice(1);
                for (int h = c.headStart(); h < c.headEnd(); h++) {
                    float a = flashAttnHeads[i][h];
                    if (a > flashAttn_m.get(h)) {
                        float e = (float) Math.exp(flashAttn_m.get(h) - a);
                        TensorOperationsProvider.get().sxpby(e, vv, value, c.maybeMapToGroupHead(h) * c.headSize, h * c.headSize, c.headSize);
                        flashAttn_l.set(1 + e * flashAttn_l.get(h), h);
                        flashAttn_m.set(a, h);
                    } else {
                        float e = (float) Math.exp(a - flashAttn_m.get(h));
                        TensorOperationsProvider.get().saxpy(e, vv, value, c.maybeMapToGroupHead(h) * c.headSize, h * c.headSize, c.headSize);
                        flashAttn_l.set(flashAttn_l.get(h) + e, h);
                    }
                }
            }

            // scale y by 1/l
            for (int h = c.headStart(); h < c.headEnd(); h++) {
                float scale = 1.0f / flashAttn_l.get(h);
                TensorOperationsProvider.get().scale(scale, value, (h * c.headSize), c.headSize);
            }

            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            AbstractTensor result = m.makeFullTensor(c.embeddingLength);
            try(AbstractTensor vq = m.maybeQuantize(value)) {
                VectorMath.pchunk(0, c.embeddingLength, (chunkStart, chunkSize) -> {
                    TensorOperationsProvider.get().dotProductChunk(result, vq, outputProjectionWeights, c.embeddingSegmentStart(), c.embeddingSegmentLength(), chunkStart, chunkSize);
                });

                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));

                outputProjectionBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(result, bias, c.embeddingSegmentStart(), c.embeddingSegmentLength()));
            }

            return result;
        }
    }
}
