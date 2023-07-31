package com.github.tjake.llmj.safetensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.primitives.Ints;

import java.util.Arrays;

public class TensorInfo {

    @JsonProperty("dtype")
    public final DType dType;

    @JsonProperty("shape")
    public final int[] shape;

    @JsonProperty("data_offsets")
    public final long[] dataOffsets;

    @JsonCreator
    public TensorInfo(@JsonProperty("dtype") DType dType,
                      @JsonProperty("shape") long[] shape,
                      @JsonProperty("data_offsets") long[] dataOffsets)
    {
        this.dType = dType;
        this.shape = new int[shape.length];
        for (int i = 0; i < shape.length; i++)
            this.shape[i] = Ints.checkedCast(shape[i]);
        this.dataOffsets = dataOffsets;
    }

    @Override
    public String toString() {
        return "TensorInfo{" +
                "dType=" + dType +
                ", shape=" + Arrays.toString(shape) +
                ", dataOffsets=" + Arrays.toString(dataOffsets) +
                "}";
    }
}
