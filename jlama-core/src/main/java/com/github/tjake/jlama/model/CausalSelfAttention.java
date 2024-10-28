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

import static com.github.tjake.jlama.util.DebugSupport.debug;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.google.common.base.Preconditions;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Consumer;

public class CausalSelfAttention {
    private static final Logger logger = LoggerFactory.getLogger(CausalSelfAttention.class);

    private final AbstractModel m;
    private final Config c;
    private final int layerIndex;
    private final DistributedContext dctx;
    private final Optional<AbstractTensor> queryAttnBias;
    private final Optional<AbstractTensor> keyAttnBias;

    private final Optional<AbstractTensor> valueAttnBias;
    private final Optional<AbstractTensor> outputProjectionBias;

    final AbstractTensor queryAttnWeights;
    final AbstractTensor keyAttnWeights;

    final AbstractTensor valueAttnWeights;

    private final AbstractTensor outputProjectionWeights;

    private final float attentionScale;
    private final int attentionLength;

    private final AbstractTensor[] qkvResults;
    private final AbstractTensor[] qkvWeights;

    public CausalSelfAttention(
        AbstractModel m,
        int layerIndex,
        AbstractTensor queryAttnWeights,
        AbstractTensor keyAttnWeights,
        AbstractTensor valueAttnWeights,
        AbstractTensor outputProjectionWeights
    ) {
        this(
            m,
            layerIndex,
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            queryAttnWeights,
            keyAttnWeights,
            valueAttnWeights,
            Optional.empty(),
            outputProjectionWeights
        );
    }

    public CausalSelfAttention(
        AbstractModel m,
        int layerIndex,
        AbstractTensor queryAttnBias,
        AbstractTensor keyAttnBias,
        AbstractTensor valueAttnBias,
        AbstractTensor queryAttnWeights,
        AbstractTensor keyAttnWeights,
        AbstractTensor valueAttnWeights,
        AbstractTensor outputProjectionBias,
        AbstractTensor outputProjectionWeights
    ) {
        this(
            m,
            layerIndex,
            Optional.of(queryAttnBias),
            Optional.of(keyAttnBias),
            Optional.of(valueAttnBias),
            queryAttnWeights,
            keyAttnWeights,
            valueAttnWeights,
            Optional.of(outputProjectionBias),
            outputProjectionWeights
        );
    }

    public CausalSelfAttention(
        AbstractModel m,
        int layerIndex,
        Optional<AbstractTensor> queryAttnBias,
        Optional<AbstractTensor> keyAttnBias,
        Optional<AbstractTensor> valueAttnBias,
        AbstractTensor queryAttnWeights,
        AbstractTensor keyAttnWeights,
        AbstractTensor valueAttnWeights,
        Optional<AbstractTensor> outputProjectionBias,
        AbstractTensor outputProjectionWeights
    ) {
        this.m = m;
        this.layerIndex = layerIndex;
        this.c = m.c;
        this.dctx = m.c.dctx();
        this.queryAttnBias = queryAttnBias;
        this.keyAttnBias = keyAttnBias;
        this.valueAttnBias = valueAttnBias;
        this.queryAttnWeights = queryAttnWeights;
        this.keyAttnWeights = keyAttnWeights;
        this.valueAttnWeights = valueAttnWeights;

        this.outputProjectionBias = outputProjectionBias;
        this.outputProjectionWeights = outputProjectionWeights;
        this.attentionLength = c.numberOfHeads * c.headSize;

        this.attentionScale = c.attentionMultiplier != null ? c.attentionMultiplier : (float) (1.0 / StrictMath.sqrt(c.headSize));

        this.qkvResults = new AbstractTensor[3];
        this.qkvWeights = new AbstractTensor[] { queryAttnWeights, keyAttnWeights, valueAttnWeights };
    }

    public AbstractTensor forward(
        AbstractTensor input,
        int startPosition,
        KvBufferCache.KvBuffer kvMem,
        Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        Preconditions.checkArgument(input.dims() == 2 && input.shape().last() == c.embeddingLength);
        int batchSize = input.shape().first();

        try (
            AbstractTensor queryBatch = m.makeDenseTensor(batchSize, attentionLength);
            AbstractTensor tmpKeyBatch = m.makeDenseTensor(batchSize, c.kvLength);
            AbstractTensor tmpValBatch = m.makeDenseTensor(batchSize, c.kvLength);
            AbstractTensor valueBatch = m.makeDenseTensor(batchSize, attentionLength)
        ) {

            if (c.isGQA) {
                VectorMath.pchunk(dctx.attentionSegmentStart, dctx.attentionSegmentLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get()
                        .dotProductChunk(queryBatch, input, queryAttnWeights, 0, c.embeddingLength, chunkStart, chunkLength);
                });
                VectorMath.pchunk(dctx.kvSegmentStart, dctx.kvSegmentLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get()
                        .dotProductChunk(tmpKeyBatch, input, keyAttnWeights, 0, c.embeddingLength, chunkStart, chunkLength);
                    TensorOperationsProvider.get()
                        .dotProductChunk(tmpValBatch, input, valueAttnWeights, 0, c.embeddingLength, chunkStart, chunkLength);
                });
            } else {
                qkvResults[0] = queryBatch;
                qkvResults[1] = tmpKeyBatch;
                qkvResults[2] = tmpValBatch;

                // compute the query vector
                VectorMath.pchunk(dctx.attentionSegmentStart, dctx.attentionSegmentLength, (chunkStart, chunkLength) -> {
                    TensorOperationsProvider.get()
                        .dotProductBatchChunk(qkvResults, input, qkvWeights, 0, c.embeddingLength, chunkStart, chunkLength);
                });
            }

            queryAttnBias.ifPresent(
                bias -> TensorOperationsProvider.get().accumulate(queryBatch, bias, dctx.attentionSegmentStart, dctx.attentionSegmentLength)
            );
            keyAttnBias.ifPresent(
                bias -> TensorOperationsProvider.get().accumulate(tmpKeyBatch, bias, dctx.kvSegmentStart, dctx.kvSegmentLength)
            );
            valueAttnBias.ifPresent(
                bias -> TensorOperationsProvider.get().accumulate(tmpValBatch, bias, dctx.kvSegmentStart, dctx.kvSegmentLength)
            );

            debug("query", queryBatch, layerIndex);
            debug("key", tmpKeyBatch, layerIndex);
            debug("value", tmpValBatch, layerIndex);

            // This is our memory of the key and value vectors for each position
            for (int position = startPosition, bi = 0; position < startPosition + batchSize; position++, bi++) {
                int finalPostion = position;
                AbstractTensor key = kvMem.getKeyTensorForPosition(layerIndex, position);
                AbstractTensor val = kvMem.getValTensorForPosition(layerIndex, position);

                AbstractTensor[] kvp = kvMem.getKeyTensorsUptoPosition(layerIndex, position);
                AbstractTensor[] vvp = kvMem.getValTensorsUptoPosition(layerIndex, position);

                AbstractTensor tmpKey = tmpKeyBatch.slice(bi);
                AbstractTensor tmpVal = tmpValBatch.slice(bi);
                AbstractTensor query = queryBatch.slice(bi);
                AbstractTensor value = valueBatch.slice(bi);

                if (key.dType() != tmpKey.dType()) {
                    try (
                        AbstractTensor tmpKey2 = TensorOperationsProvider.get().quantize(tmpKey, key.dType(), 0, c.kvLength);
                        AbstractTensor tmpVal2 = TensorOperationsProvider.get().quantize(tmpVal, val.dType(), 0, c.kvLength)
                    ) {
                        key.copyFrom(
                            tmpKey2,
                            tmpKey2.getOffset(0, dctx.kvSegmentStart),
                            key.getOffset(0, dctx.kvSegmentStart),
                            dctx.kvSegmentLength
                        );
                        val.copyFrom(
                            tmpVal2,
                            tmpVal2.getOffset(0, dctx.kvSegmentStart),
                            val.getOffset(0, dctx.kvSegmentStart),
                            dctx.kvSegmentLength
                        );
                    }
                } else {
                    key.copyFrom(
                        tmpKey,
                        tmpKey.getOffset(0, dctx.kvSegmentStart),
                        key.getOffset(0, dctx.kvSegmentStart),
                        dctx.kvSegmentLength
                    );
                    val.copyFrom(
                        tmpVal,
                        tmpVal.getOffset(0, dctx.kvSegmentStart),
                        val.getOffset(0, dctx.kvSegmentStart),
                        dctx.kvSegmentLength
                    );
                }

                // apply RoPE if present (accounting for huggingface permutation)
                // https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/models/llama/convert_llama_weights_to_hf.py#L114
                c.ropeFreqs.ifPresent(rf -> {
                    int headPiece = c.headSize / 2;
                    int poffset = finalPostion * headPiece;

                    if (c.isGQA) {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = dctx.headStart; h < dctx.headEnd; h++) {
                            // get the q vectors for this head
                            int offset = h * c.headSize;

                            // skip if we are out of bounds
                            if (offset >= query.shape().last()) break;

                            int goffset = c.maybeMapToGroupHead(h) * c.headSize;
                            // rotate q by the freq theta and freq r
                            for (int i = offset, g = goffset; i < (offset + headPiece); i++, g++) {
                                float q0 = query.get(0, i);
                                float q1 = query.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float[] f = rf[poffset + g];
                                float fcr = f[0];
                                float fci = f[1];
                                query.set(q0 * fcr - q1 * fci, 0, i);
                                query.set(q0 * fci + q1 * fcr, 0, i + headPiece);
                            }
                        }

                        for (int h = dctx.groupHeadStart; h < dctx.groupHeadEnd; h++) {
                            // get the k vectors for this head
                            int offset = h * c.headSize;
                            if (offset >= key.shape().last()) break;
                            // rotate k by the freq theta and freq r
                            for (int i = offset; i < (offset + headPiece); i++) {
                                float k00 = key.get(0, i);
                                float k1 = key.get(0, i + headPiece); // hf permutation is 0,64,1,65 etc...
                                float[] f = rf[poffset + i];
                                float fcr = f[0];
                                float fci = f[1];
                                key.set(k00 * fcr - k1 * fci, 0, i);
                                key.set(k00 * fci + k1 * fcr, 0, i + headPiece);
                            }
                        }
                    } else {
                        // apply RoPE rotation to the q and k vectors for each head
                        for (int h = dctx.headStart; h < dctx.headEnd; h++) {
                            // get the q and k vectors for this head
                            int offset = h * c.headSize;
                            // rotate q and k by the freq theta and freq r
                            for (int i = offset; i < (offset + headPiece); i++) {
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
                    debug("query+rope", query, finalPostion);
                    debug("key+rope", key, finalPostion);
                });

                // Attention
                VectorMath.pfor(dctx.headStart, dctx.headEnd, h -> {
                    int xoffset = c.maybeMapToGroupHead(h) * c.headSize;
                    int yoffset = h * c.headSize;

                    if (yoffset >= query.shape().last()) return;

                    try (AbstractTensor attn = m.makeDenseTensor(1, kvp[0].shape().first() * kvp.length)) { // chunky so the cache isn't
                                                                                                            // thrashed
                        // compute attention scores by multiplying query and key for every position
                        // Do this for each page
                        for (int i = 0; i < kvp.length; i++) {
                            int len = kvp[i].shape().first();
                            int offset = i * len;
                            int size = i == kvp.length - 1 ? (finalPostion + 1) - offset : len;
                            TensorOperationsProvider.get()
                                .batchDotProduct(attn, query, kvp[i], yoffset, xoffset, c.headSize, offset, 0, size);
                        }

                        TensorOperationsProvider.get().scale(attentionScale, attn, 0, finalPostion + 1);

                        if (c.attnLogitSoftCapping != null) {
                            for (int i = 0; i < finalPostion + 1; i++) {
                                float v = attn.get(0, i);
                                v /= c.attnLogitSoftCapping;
                                v = (float) FastMath.tanh(v);
                                v *= c.attnLogitSoftCapping;
                                attn.set(v, 0, i);
                            }
                        }

                        // softmax the scores to get attention weights, from 0..pos inclusively
                        VectorMath.softMax(attn, 0, finalPostion + 1);

                        // apply adjusted attention weights to value vectors
                        // do this for each page
                        for (int i = 0; i < vvp.length; i++) {
                            int len = vvp[i].shape().first();
                            int offset = i * len;
                            int size = i == vvp.length - 1 ? (finalPostion + 1) - offset : len;
                            TensorOperationsProvider.get().saxpy(attn, vvp[i], value, xoffset, yoffset, c.headSize, offset, 0, size);
                        }
                    }
                });
            }

            debug("after_attention", valueBatch, layerIndex);

            // matmul the projection and sum into input
            // input += c_proj_weight @ ybuf + c_proj_bias
            AbstractTensor result = m.makeDenseTensor(batchSize, c.embeddingLength);
            try (AbstractTensor vq = m.maybeQuantize(valueBatch)) {
                VectorMath.pchunk(0, c.embeddingLength, (chunkStart, chunkSize) -> {
                    TensorOperationsProvider.get()
                        .dotProductChunk(
                            result,
                            vq,
                            outputProjectionWeights,
                            dctx.attentionSegmentStart,
                            dctx.attentionSegmentLength,
                            chunkStart,
                            chunkSize
                        );
                });

                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));

                outputProjectionBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(result, bias, 0, c.embeddingLength));
            }

            return result;
        }
    }
}
