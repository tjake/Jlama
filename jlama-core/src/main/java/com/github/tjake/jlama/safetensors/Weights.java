package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.tensor.*;

import com.github.tjake.jlama.util.Pair;
import com.google.common.collect.ImmutableMap;
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
    private final DType majorityDType;

    Weights(Map<String, String> metadata, Map<String, TensorInfo> tensorInfoMap, ByteBuffer bytes)
    {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.tensorInfoMap = ImmutableMap.copyOf(tensorInfoMap);
        this.bytes = bytes.duplicate();
        this.majorityDType = findDType();
    }

    private DType findDType() {
        EnumMap<DType, Integer> counts = new EnumMap<>(DType.class);
        for (Map.Entry<String, TensorInfo> e : tensorInfoMap.entrySet()) {
            if (!e.getKey().endsWith(".qb"))
                counts.put(e.getValue().dType, counts.getOrDefault(e.getValue().dType, 0) + 1);
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
        return maxType == DType.BF16 || maxType == DType.F16 ? DType.F32 : maxType;
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
    public AbstractTensor load(String name, Optional<Pair<Integer, Integer>> offset) throws NoSuchElementException {
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
        AbstractTensor t;
        switch (info.dType) {
            case F32:
                fb = b.asFloatBuffer().slice();
                t =  new FloatBufferTensor(name, fb, TensorShape.of(info.shape), true);
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
                    t =  new FloatBufferTensor(bb.asFloatBuffer(), TensorShape.of(info.shape), true);
                } else {
                    sb = b.asShortBuffer().slice();
                    t =  new Float16BufferTensor(name, sb, TensorShape.of(info.shape), true);
                }
                break;
            case BF16:
                //For now always convert to F32
                len = b.remaining() / DType.F16.size();
                fb = FloatBuffer.allocate(len);
                for (int i = 0; i < len; i++) {
                    short s = b.getShort();
                    float v = FloatConversions.bFloat16ToFloat32(s);
                    fb.put(i, v);
                }
                t = new FloatBufferTensor(name, fb, TensorShape.of(info.shape), true);
                break;
            case Q4:
                FloatBufferTensor qb = (FloatBufferTensor) load(name + ".qb", offset);
                t = new Q4ByteBufferTensor(name, b.slice(), qb, TensorShape.of(info.shape), true);
                break;
            case I8:
                FloatBufferTensor qb1 = (FloatBufferTensor) load(name + ".qb", offset);
                t = new Q8ByteBufferTensor(name, b.slice(), qb1, TensorShape.of(info.shape), true);
                break;
            default:
                throw new IllegalArgumentException("Unsupported Tensor type: " + info.dType.name() + " for " + name);
        }

        return offset.map(o -> t.sparsify(o.left, o.right)).orElse(t);
    }

    @Override
    public DType getModelDType() {
        return majorityDType;
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

    @Override
    public void close() throws Exception {

    }
}
