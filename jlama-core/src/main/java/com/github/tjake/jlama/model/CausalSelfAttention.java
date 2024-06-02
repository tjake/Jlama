/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.TensorCache;
import com.github.tjake.jlama.tensor.operations.TensorOperations;
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

    private final AbstractTensor[] qkvResults;
    private final AbstractTensor[] qkvWeights;

    private static final boolean USE_FLASH_ATTN = false;


    public CausalSelfAttention(
            AbstractModel m,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            AbstractTensor outputProjectionWeights) {
        this(
                m,
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                queryAttnWeights,
                keyAttnWeights,
                valueAttnWeights,
                Optional.empty(),
                outputProjectionWeights);
    }

    public CausalSelfAttention(
            AbstractModel m,
            AbstractTensor queryAttnBias,
            AbstractTensor keyAttnBias,
            AbstractTensor valueAttnBias,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            AbstractTensor outputProjectionBias,
            AbstractTensor outputProjectionWeights) {
        this(
                m,
                Optional.of(queryAttnBias),
                Optional.of(keyAttnBias),
                Optional.of(valueAttnBias),
                queryAttnWeights,
                keyAttnWeights,
                valueAttnWeights,
                Optional.of(outputProjectionBias),
                outputProjectionWeights);
    }

    public CausalSelfAttention(
            AbstractModel m,
            Optional<AbstractTensor> queryAttnBias,
            Optional<AbstractTensor> keyAttnBias,
            Optional<AbstractTensor> valueAttnBias,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            Optional<AbstractTensor> outputProjectionBias,
            AbstractTensor outputProjectionWeights) {
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

        this.qkvResults = new AbstractTensor[3];
        this.qkvWeights = new AbstractTensor[] {queryAttnWeights, keyAttnWeights, valueAttnWeights};
    }

    public AbstractTensor forward(
            AbstractTensor input,
            int startPosition,
            AbstractTensor kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == c.embeddingLength);
        int batchSize = input.shape().first();

        try (AbstractTensor flashAttn_m = m.makeTensor(batchSize, c.numberOfHeads);
                AbstractTensor flashAttn_l = m.makeTensor(batchSize, c.numberOfHeads);
                AbstractTensor queryBatch = m.makeFullTensor(batchSize, c.embeddingLength);
                AbstractTensor tmpKeyBatch = m.makeFullTensor(batchSize, c.kvLength);
                AbstractTensor tmpValBatch = m.makeFullTensor(batchSize, c.kvLength);
                AbstractTensor valueBatch = m.makeFullTensor(batchSize, c.embeddingLength)) {


            if (c.isGQA) {
                VectorMath.pchunk(0, c.embeddingLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get()
                            .dotProductChunk(
                                    queryBatch,
                                    input,
                                    queryAttnWeights,
                                    c.embeddingSegmentStart(),
                                    c.embeddingSegmentLength(),
                                    chunkStart,
                                    chunkLength);
                });
                VectorMath.pchunk(0, c.kvLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get()
                            .dotProductChunk(
                                    tmpKeyBatch,
                                    input,
                                    keyAttnWeights,
                                    c.embeddingSegmentStart(),
                                    c.embeddingSegmentLength(),
                                    chunkStart,
                                    chunkLength);
                    TensorOperationsProvider.get()
                            .dotProductChunk(
                                    tmpValBatch,
                                    input,
                                    valueAttnWeights,
                                    c.embeddingSegmentStart(),
                                    c.embeddingSegmentLength(),
                                    chunkStart,
                                    chunkLength);
                });
            } else {
                qkvResults[0] = queryBatch;
                qkvResults[1] = tmpKeyBatch;
                qkvResults[2] = tmpValBatch;

                // compute the query vector
                VectorMath.pchunk(0, c.embeddingLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get()
                            .dotProductBatchChunk(
                                    qkvResults,
                                    input,
                                    qkvWeights,
                                    c.embeddingSegmentStart(),
                                    c.embeddingSegmentLength(),
                                    chunkStart,
                                    chunkLength);
                });
            }

            // For distributed sum of tensor
            tensorReducer.ifPresent(func -> func.accept(List.of(queryBatch, tmpKeyBatch, tmpValBatch)));

            queryAttnBias.ifPresent(bias ->
                    TensorOperationsProvider.get().accumulate(queryBatch, bias, c.embeddingSegmentStart(), c.embeddingSegmentLength()));
            keyAttnBias.ifPresent(bias ->
                    TensorOperationsProvider.get().accumulate(tmpKeyBatch, bias, c.kvSegmentStart(), c.kvSegmentLength()));
            valueAttnBias.ifPresent(bias ->
                    TensorOperationsProvider.get().accumulate(tmpValBatch, bias, c.kvSegmentStart(), c.kvSegmentLength()));


            // with all key-value entries populated, compute attention
            // the softmax is incrementally aggregated using the flash attention technique
            AbstractTensor k0 = kvMem.slice(true, 0).slice(0);
            AbstractTensor v0 = kvMem.slice(true, 1).slice(0);


            // This is our memory of the key and value vectors for each position
            for (int position = startPosition, bi = 0; position < startPosition + batchSize; position++, bi++)
            {
                int finalPostion = position;
                int finalBi = bi;

                AbstractTensor kvp = kvMem.slice(true, 0);
                AbstractTensor vvp = kvMem.slice(true, 1);

                AbstractTensor key = kvp.slice(position);
                AbstractTensor val = vvp.slice(position);

                AbstractTensor tmpKey = tmpKeyBatch.slice(bi);
                AbstractTensor tmpVal = tmpValBatch.slice(bi);
                AbstractTensor query = queryBatch.slice(bi);
                AbstractTensor value = valueBatch.slice(bi);

                key.copyFrom(
                        tmpKey,
                        tmpKey.getOffset(0, c.kvSegmentStart()),
                        key.getOffset(0, c.kvSegmentStart()),
                        c.kvSegmentLength());
                val.copyFrom(
                        tmpVal,
                        tmpVal.getOffset(0, c.kvSegmentStart()),
                        val.getOffset(0, c.kvSegmentStart()),
                        c.kvSegmentLength());

                // apply RoPE if present (accounting for huggingface permutation)
                // https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/models/llama/convert_llama_weights_to_hf.py#L114
                c.ropeFreqs.ifPresent(rf -> {
                    int headPiece = c.headSize / 2;
                    int poffset = finalPostion * headPiece;

                    if (c.isGQA)
                    {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = c.headStart(); h < c.headEnd(); h++)
                        {
                            // get the q vectors for this head
                            int offset = h * c.headSize;
                            int goffset = c.maybeMapToGroupHead(h) * c.headSize;
                            // rotate q by the freq theta and freq r
                            for (int i = offset, g = goffset; i < (offset + headPiece); i++, g++)
                            {
                                float q0 = query.get(0, i);
                                float q1 = query.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float[] f = rf[poffset + g];
                                float fcr = f[0];
                                float fci = f[1];
                                query.set(q0 * fcr - q1 * fci, 0, i);
                                query.set(q0 * fci + q1 * fcr, 0, i + headPiece);
                            }
                        }

                        for (int h = c.groupHeadStart(); h < c.groupHeadEnd(); h++)
                        {
                            // get the k vectors for this head
                            int offset = h * c.headSize;
                            // rotate k by the freq theta and freq r
                            for (int i = offset; i < (offset + headPiece); i++)
                            {
                                float k00 = key.get(0, i);
                                float k1 = key.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float[] f = rf[poffset + i];
                                float fcr = f[0];
                                float fci = f[1];
                                key.set(k00 * fcr - k1 * fci, 0, i);
                                key.set(k00 * fci + k1 * fcr, 0, i + headPiece);
                            }
                        }
                    }
                    else
                    {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = c.headStart(); h < c.headEnd(); h++)
                        {
                            // get the q and k vectors for this head
                            int offset = h * c.headSize;
                            // rotate q and k by the freq theta and freq r
                            for (int i = offset; i < (offset + headPiece); i++)
                            {
                                float q0 = query.get(0, i);
                                float q1 = query.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float k00 = key.get(0, i);
                                float k1 = key.get(0, i + headPiece);
                                float[] f = rf[poffset + i];
                                float fcr = f[0];
                                float fci = f[1];
                                query.set(q0 * fcr - q1 * fci, 0, i);
                                query.set(q0 * fci + q1 * fcr, 0, i + headPiece);
                                key.set(k00 * fcr - k1 * fci, 0, i);
                                key.set(k00 * fci + k1 * fcr, 0, i + headPiece);
                            }
                        }
                    }
                });

                if (USE_FLASH_ATTN)
                {

                    // value is initially the position 0 value for all heads
                    // POSITION ZERO
                    for (int i = c.headStart(); i < c.headEnd(); i++)
                    {
                        value.copyFrom(
                                v0,
                                v0.getOffset(0, c.maybeMapToGroupHead(i) * c.headSize),
                                value.getOffset(0, i * c.headSize),
                                c.headSize);
                        float a = TensorOperationsProvider.get()
                                .dotProduct(
                                        query,
                                        k0,
                                        i * c.headSize,
                                        c.maybeMapToGroupHead(i) * c.headSize,
                                        c.headSize)
                                * attentionScale;
                        flashAttn_m.set(a, bi, i);
                        flashAttn_l.set(1, bi, i);
                    }

                    // POSITION > 0
                    // This is where the context length gets expensive! We need to run this query token by all prior tokens.
                    // Now aggregate results per head
                    VectorMath.pfor(c.headStart(), c.headEnd(), h -> {
                        int xoffset = c.maybeMapToGroupHead(h) * c.headSize;
                        int yoffset = h * c.headSize;
                        for (int i = 0; i < finalPostion; i++)
                        {

                            AbstractTensor pmem = kvMem.slice(true, i + 1);
                            // KEY
                            float a = TensorOperationsProvider.get()
                                    .dotProduct(
                                            query,
                                            pmem,
                                            yoffset,
                                            xoffset,
                                            c.headSize)
                                    * attentionScale;

                            // VALUE
                            if (a > flashAttn_m.get(finalBi, h))
                            {
                                float e = (float) Math.exp(flashAttn_m.get(finalBi, h) - a);
                                TensorOperationsProvider.get()
                                        .sxpby(
                                                e,
                                                pmem,
                                                value,
                                                xoffset,
                                                yoffset,
                                                c.headSize);
                                flashAttn_l.set(1 + e * flashAttn_l.get(finalBi, h), finalBi, h);
                                flashAttn_m.set(a, finalBi, h);
                            }
                            else
                            {
                                float e = (float) Math.exp(a - flashAttn_m.get(finalBi, h));
                                TensorOperationsProvider.get()
                                        .saxpy(
                                                e,
                                                pmem,
                                                value,
                                                pmem.getOffset(1, xoffset),
                                                yoffset,
                                                c.headSize);
                                flashAttn_l.set(flashAttn_l.get(finalBi, h) + e, finalBi, h);
                            }
                        }
                    });

                    // scale y by 1/l
                    for (int h = c.headStart(); h < c.headEnd(); h++)
                    {
                        float scale = 1.0f / flashAttn_l.get(bi, h);
                        TensorOperationsProvider.get().scale(scale, value, (h * c.headSize), c.headSize);
                    }
                }
                else
                {

                    VectorMath.pfor(c.headStart(), c.headEnd(), h -> {
                        try (AbstractTensor attn = m.makeFullTensor(1, kvp.shape().first()))
                        {
                            int xoffset = c.maybeMapToGroupHead(h) * c.headSize;
                            int yoffset = h * c.headSize;

                            // compute attention scores by multiplying query and key for every position
                            TensorOperationsProvider.get().batchDotProduct(attn, query, kvp, yoffset, xoffset, c.headSize, 0, finalPostion + 1);
                            TensorOperationsProvider.get().scale(attentionScale, attn, 0, finalPostion + 1);

                            // softmax the scores to get attention weights, from 0..pos inclusively
                            VectorMath.softMax(attn, 0, finalPostion + 1);

                            // apply adjusted attention weights to value vectors
                            TensorOperationsProvider.get().saxpy(attn, vvp, value, xoffset, yoffset, c.headSize, finalPostion + 1);
                        }
                    });

                }
            }
            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            AbstractTensor result = m.makeFullTensor(batchSize, c.embeddingLength);
            try (AbstractTensor vq = m.maybeQuantize(valueBatch)) {
                VectorMath.pchunk(0, c.embeddingLength, (chunkStart, chunkSize) -> {
                    TensorOperationsProvider.get()
                            .dotProductChunk(
                                    result,
                                    vq,
                                    outputProjectionWeights,
                                    c.embeddingSegmentStart(),
                                    c.embeddingSegmentLength(),
                                    chunkStart,
                                    chunkSize);
                });

                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));

                outputProjectionBias.ifPresent(bias -> TensorOperationsProvider.get()
                        .accumulate(result, bias, c.embeddingSegmentStart(), c.embeddingSegmentLength()));
            }

            return result;
        }
    }
}
