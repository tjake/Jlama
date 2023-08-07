package com.github.tjake.jlama.model;

import jdk.incubator.vector.FloatVector;

import java.nio.Buffer;

/** A Tensor is a multi-dimensional array of floats.
 * See {@link FloatBufferTensor} for primary impl.
 *
 * This is a interface because there could be multiple implementations one day for 8bit floats?
 **/
public interface Tensor extends AutoCloseable {

    /** Number of dimensions */
    int dims();

    /** Represents the dimensions of the tensor */
    int[] shape();

    /** Total capacity of the tensor */
    int size();

    /** Get a value at the given coordinates */
    float get(int ...dims);

    /** Set a value at the given coordinates */
    void set(float v, int ...dims);

    Tensor slice(int ...dims);

    /** Split the tensor into numChunks along the given dimension */
    Tensor[] split(int numChunks, int dim);

    Tensor transpose();
    Buffer getBuffer();
    float[] getFloatArray();

    FloatVector getVector(int offset);

    int getArrayOffset();

    boolean hasArray();

    default void close() {}

    default void debug(String id) {
        if (false) {
            double tmp = 0.0;
            for (int i = 0; i < size(); i++) {
                tmp += get(i);
            }
            System.out.println(String.format("%s = %.5f", id, tmp));
        }
    }
}
