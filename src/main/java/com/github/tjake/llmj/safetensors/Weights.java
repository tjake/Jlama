package com.github.tjake.llmj.safetensors;

import com.github.tjake.llmj.model.FloatBufferTensor;
import com.google.common.primitives.Ints;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

public class Weights {

    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> tensorInfoMap;
    private final ByteBuffer bytes;

    Weights(Map<String, String> metadata, Map<String, TensorInfo> tensorInfoMap, ByteBuffer bytes)
    {
        this.metadata = metadata;
        this.tensorInfoMap = tensorInfoMap;
        this.bytes = bytes.duplicate();
    }

    public FloatBufferTensor load(String name) throws NoSuchElementException {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null)
            throw new NoSuchElementException();

        if (info.shape.length < 1)
            throw new RuntimeException("Invalid shape dimensions " + info.shape.length + " encountered for " + name);

        ByteBuffer b =  bytes.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                .position(Ints.checkedCast(info.dataOffsets[0]))
                .limit(Ints.checkedCast(info.dataOffsets[1]));

        FloatBuffer fb;
        if (info.dType == DType.F32) {
            fb = b.asFloatBuffer().slice();
        } else if( info.dType == DType.F16) {
            int len = b.remaining() / DType.F16.size();
            fb = FloatBuffer.allocate(len); //.allocateDirect(b.remaining() * DType.F16.size()).asFloatBuffer();
            for (int i = 0; i < len; i++) {
                short s = b.getShort();
                float v = Float.float16ToFloat(s);
                fb.put(i,v);
            }
        } else {
            throw new UnsupportedOperationException();
        }

        return new FloatBufferTensor(fb, info.shape, true);
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
