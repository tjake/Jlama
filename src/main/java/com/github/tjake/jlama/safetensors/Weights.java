package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.model.AbstractTensor;
import com.github.tjake.jlama.model.Float16BufferTensor;
import com.github.tjake.jlama.model.FloatBufferTensor;
import com.google.common.primitives.Ints;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.*;

public class Weights implements WeightLoader {
    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> tensorInfoMap;
    private final ByteBuffer bytes;
    private final DType dType;

    Weights(Map<String, String> metadata, Map<String, TensorInfo> tensorInfoMap, ByteBuffer bytes)
    {
        this.metadata = metadata;
        this.tensorInfoMap = tensorInfoMap;
        this.bytes = bytes.duplicate();
        this.dType = findDType();
    }

    private DType findDType() {
        EnumMap<DType, Integer> counts = new EnumMap<>(DType.class);
        for (TensorInfo info : tensorInfoMap.values()) {
            counts.put(info.dType, counts.getOrDefault(info.dType, 0) + 1);
        }

        int max = 0;
        DType maxType = null;
        for (Map.Entry<DType, Integer> e : counts.entrySet()) {
            if (e.getValue() > max) {
                max = e.getValue();
                maxType = e.getKey();
            }
        }

        //FIXME don't really support B16 atm
        return maxType == DType.BF16 ? DType.F32 : maxType;
    }

    @Override
    public AbstractTensor load(String name) throws NoSuchElementException {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null)
            throw new NoSuchElementException();

        if (info.shape.length < 1)
            throw new RuntimeException("Invalid shape dimensions " + info.shape.length + " encountered for " + name);

        ByteBuffer b = bytes.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                .position(Ints.checkedCast(info.dataOffsets[0]))
                .limit(Ints.checkedCast(info.dataOffsets[1]));

        int len;
        FloatBuffer fb;
        ShortBuffer sb;
        switch (info.dType) {
            case F32:
                fb = b.asFloatBuffer().slice();
                return new FloatBufferTensor(fb, info.shape, true, true);
            case F16:
                // If the majority of the weights are F32 then convert to F32
                if (dType == DType.F32) {
                    len = b.remaining() / DType.F16.size();
                    ByteBuffer bb = ByteBuffer.allocateDirect(len * DType.F32.size()).order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < len * DType.F32.size(); i += DType.F32.size()) {
                        short s = b.getShort();
                        float v = Float.float16ToFloat(s);
                        bb.putFloat(i, v);
                    }
                    return new FloatBufferTensor(bb.asFloatBuffer(), info.shape, true, false);
                } else {
                    sb = b.asShortBuffer().slice();
                    return new Float16BufferTensor(sb, info.shape, true, true);
                }
            case BF16:
                //For now always convert to F32
                len = b.remaining() / DType.F16.size();
                fb = FloatBuffer.allocate(len);
                for (int i = 0; i < len; i++) {
                    short s = b.getShort();
                    float v = FloatConversions.bFloat16ToFloat32(s);
                    fb.put(i, v);
                }
                return new FloatBufferTensor(fb, info.shape, true, false);
            default:
                throw new IllegalArgumentException("Unsupported Tensor type: " + info.dType.name() + " for " + name);
        }
    }

    @Override
    public DType getModelDType() {
        return dType;
    }

    @Override
    public String toString() {
        return "SafeTensor{" +
                "metadata=" + metadata +
                ", tensorInfoMap=" + tensorInfoMap +
                ", bytes=" + bytes +
                '}';
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

}
