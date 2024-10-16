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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.primitives.Ints;
import java.util.Arrays;
import java.util.Objects;

public class TensorInfo implements Comparable<TensorInfo> {

    @JsonProperty("dtype")
    public final DType dType;

    @JsonProperty("shape")
    public final int[] shape;

    @JsonProperty("data_offsets")
    public final long[] dataOffsets;

    @JsonCreator
    public TensorInfo(
        @JsonProperty("dtype") DType dType,
        @JsonProperty("shape") long[] shape,
        @JsonProperty("data_offsets") long[] dataOffsets
    ) {
        this.dType = dType;
        this.shape = new int[shape.length];
        for (int i = 0; i < shape.length; i++)
            this.shape[i] = Ints.checkedCast(shape[i]);
        this.dataOffsets = dataOffsets;
    }

    @Override
    public String toString() {
        return "TensorInfo{"
            + "dType="
            + dType
            + ", shape="
            + Arrays.toString(shape)
            + ", dataOffsets="
            + Arrays.toString(dataOffsets)
            + "}";
    }

    @Override
    public final boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TensorInfo that)) return false;

        return dType == that.dType && Arrays.equals(shape, that.shape) && Arrays.equals(dataOffsets, that.dataOffsets);
    }

    @Override
    public int hashCode() {
        int result = Objects.hashCode(dType);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(dataOffsets);
        return result;
    }

    @Override
    public int compareTo(TensorInfo o) {
        // In the case we are reading in order of dataOffsets
        return Long.compare(dataOffsets[0], o.dataOffsets[0]);
    }
}
