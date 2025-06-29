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
package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.model.DistributedContext;
import com.github.tjake.jlama.tensor.*;
import com.github.tjake.jlama.util.Pair;
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Ints;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Weights implements WeightLoader {
    private static final Logger logger = LoggerFactory.getLogger(Weights.class);
    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> tensorInfoMap;
    private final ByteBuffer bytes;
    private final DType majorityDType;
    private final Optional<WeightLoader> parent;

    Weights(Map<String, String> metadata, Map<String, TensorInfo> tensorInfoMap, ByteBuffer bytes, Optional<WeightLoader> parent) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.tensorInfoMap = ImmutableMap.copyOf(tensorInfoMap);
        this.bytes = bytes.duplicate();
        this.majorityDType = findDType(tensorInfoMap);
        this.parent = parent;
    }

    public static DType findDType(Map<String, TensorInfo> tensorInfoMap) {
        EnumMap<DType, Integer> counts = new EnumMap<>(DType.class);
        for (Map.Entry<String, TensorInfo> e : tensorInfoMap.entrySet()) {
            if (!e.getKey().endsWith(".qb")) counts.put(e.getValue().dType, counts.getOrDefault(e.getValue().dType, 0) + 1);
        }

        int max = 0;
        DType maxType = null;
        for (Map.Entry<DType, Integer> e : counts.entrySet()) {
            if (e.getValue() > max) {
                max = e.getValue();
                maxType = e.getKey();
            }
        }

        // FIXME don't really support F16 atm
        return maxType == DType.F16 ? DType.F32 : maxType;
    }

    @Override
    public Map<String, String> metadata() {
        return metadata;
    }

    @Override
    public Map<String, TensorInfo> tensorInfoMap() {
        return tensorInfoMap;
    }

    @Override
    public AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns) {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null) throw new NoSuchElementException(name + " not found in weights");

        if (info.shape.length < 1) throw new RuntimeException("Invalid shape dimensions " + info.shape.length + " encountered for " + name);

        if (dctx != null && info.shape.length != 2) {
            throw new RuntimeException("Invalid shape dimensions " + info.shape.length + " encountered for " + name + " with offset");
        }

        Pair<TensorShape, Pair<Long, Long>> offsets = getLoadOffsets(info, dctx, sparseRows);

        ByteBuffer b = bytes.duplicate()
            .order(ByteOrder.LITTLE_ENDIAN)
            .position(Ints.checkedCast(offsets.right.left))
            .limit(Ints.checkedCast(offsets.right.right));

        return loadTensorFromBuffer(name, info.dType, majorityDType, offsets.left, b, sparseRows, sparseColumns, dctx, parent.orElse(this));
    }

    static Pair<TensorShape, Pair<Long, Long>> getLoadOffsets(TensorInfo info, DistributedContext dctx, boolean sparseRows) {
        long positionOffset = info.dataOffsets[0];
        long positionLimit = info.dataOffsets[1];
        TensorShape shape = TensorShape.of(info.shape);

        // If this is a sparse tensor, we need to fetch only the section of the tensor that is needed
        if (dctx != null && sparseRows) {
            int rows = info.shape[0];
            int columnLength = info.shape[1] * info.dType.size();

            // Hack for Q4
            if (info.dType == DType.Q4) columnLength /= 2;

            positionOffset = info.dataOffsets[0] + (dctx.getShardOffsetForLength(rows) * columnLength);
            positionLimit = positionOffset + (dctx.getShardLength(rows) * columnLength);
            shape = TensorShape.sparseRow(info.shape, Pair.of(dctx.getShardOffsetForLength(rows), dctx.getShardLength(rows)));
        }
        return Pair.of(shape, Pair.of(positionOffset, positionLimit));
    }

    static AbstractTensor loadTensorFromBuffer(
        String name,
        DType dType,
        DType majorityDType,
        TensorShape shape,
        ByteBuffer b,
        boolean sparseRows,
        boolean sparseColumns,
        DistributedContext dctx,
        WeightLoader loader
    ) {
        int len;
        FloatBuffer fb;
        ShortBuffer sb;
        AbstractTensor t;
        switch (dType) {
            case F32:
                fb = b.asFloatBuffer().slice();
                t = new FloatBufferTensor(name, fb, shape, true);
                break;
            case F16:
                // If the majority of the weights are F32 then convert to F32
                if (majorityDType == DType.F32) {
                    len = b.remaining() / DType.F16.size();
                    ByteBuffer bb = ByteBuffer.allocate(len * DType.F32.size()).order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < len * DType.F32.size(); i += DType.F32.size()) {
                        short s = b.getShort();
                        float v = Float.float16ToFloat(s);
                        bb.putFloat(i, v);
                    }
                    t = new FloatBufferTensor(bb.asFloatBuffer(), shape, true);
                } else {
                    sb = b.asShortBuffer().slice();
                    t = new Float16BufferTensor(name, sb, shape, true);
                }
                break;
            case BF16:
                sb = b.asShortBuffer().slice();
                t = new BFloat16BufferTensor(name, sb, shape, true);
                break;
            case Q4:
                FloatBufferTensor qb = (FloatBufferTensor) loader.load(name + ".qb", dctx, sparseRows, false /*only need sparsify once*/);
                t = new Q4ByteBufferTensor(name, b.slice(), qb, shape, true);
                break;
            case I8:
                FloatBufferTensor qb1 = (FloatBufferTensor) loader.load(
                    name + ".qb",
                    dctx,
                    sparseRows,
                    false /*only need to sparsify once*/
                );
                t = new Q8ByteBufferTensor(name, b.slice(), qb1, shape, true);
                break;
            default:
                throw new IllegalArgumentException("Unsupported Tensor type: " + dType.name() + " for " + name);
        }

        return dctx != null && sparseColumns && dctx.hasModelShard()
            ? t.sparsify(dctx.getShardOffsetForLength(shape.last()), dctx.getShardLength(shape.last()))
            : t;
    }

    @Override
    public DType getModelDType() {
        return majorityDType;
    }

    @Override
    public String toString() {
        return "SafeTensor{" + "metadata=" + metadata + ", tensorInfoMap=" + tensorInfoMap + ", bytes=" + bytes + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Weights weights = (Weights) o;
        return Objects.equals(metadata, weights.metadata) && Objects.equals(tensorInfoMap, weights.tensorInfoMap);
    }

    @Override
    public int hashCode() {
        return Objects.hash(metadata, tensorInfoMap);
    }

    @Override
    public void close() throws Exception {}
}
