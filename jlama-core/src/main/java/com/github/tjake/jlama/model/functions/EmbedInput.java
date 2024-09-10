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
package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.TensorCache;
import com.github.tjake.jlama.tensor.TensorShape;
import com.google.common.base.Preconditions;

/**
 * Used to define a function that maps input tokens to embeddings
 */
public interface EmbedInput {
    AbstractTensor inputTokenToEmbedding(int inputToken, int position);

    default AbstractTensor batchInputsToEmbeddings(int[] inputTokens, int startPos) {
        Preconditions.checkArgument(inputTokens.length > 0);

        AbstractTensor t = inputTokenToEmbedding(inputTokens[0], startPos);
        if (inputTokens.length == 1) return t;

        TensorShape tbs = TensorShape.of(inputTokens.length, t.shape().last());

        AbstractTensor tb = TensorCache.instance.get(t.dType(), tbs);
        tb.copyFrom(t, 0, 0, t.shape().last());
        t.close();

        VectorMath.pfor(1, inputTokens.length, i -> {
            AbstractTensor ti = inputTokenToEmbedding(inputTokens[i], startPos + i);
            tb.copyFrom(ti, 0, i * ti.shape().last(), ti.shape().last());
            ti.close();
        });

        return tb;
    }
}
