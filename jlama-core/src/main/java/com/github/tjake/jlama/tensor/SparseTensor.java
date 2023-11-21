package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.safetensors.DType;
import jdk.incubator.vector.Vector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

public class SparseTensor extends AbstractTensor {

    public static SparseTensor wrap(AbstractTensor t, int offset, int length) {
        int[] shape = Arrays.copyOf(t.shape(), t.dims());
        shape[shape.length - 1] = length;

        AbstractTensor newT = t.make(shape);

        return new SparseTensor(newT, shape, offset, length);
    }

    private  int offset;
    private  int length;

    protected SparseTensor(AbstractTensor t, int[] originShape, int offset, int length) {
        super(t.dType, originShape, false);
    }

    @Override
    protected AbstractTensor make(int... shape) {
        return null;
    }

    @Override
    protected AbstractTensor make(int heapOffset, int heapLength, int[] shape, boolean cacheSlices) {
        return null;
    }

    @Override
    public float get(int... dims) {
        return 0;
    }

    @Override
    public void set(float v, int... dims) {

    }

    @Override
    public Object getArray() {
        return null;
    }

    @Override
    public int getArrayOffset(int offset) {
        return this.offset + offset;
    }

    @Override
    public Vector<?> getVector(VectorSpecies species, int offset) {
        return null;
    }

    @Override
    public MemorySegment getMemorySegment() {
        return null;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return this.offset + offset;
    }

    @Override
    public boolean hasMemorySegment() {
        return false;
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {

    }

    @Override
    public void clear() {

    }

    @Override
    public void intoTensor(Vector vector, int offset) {

    }
}
