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
import com.github.tjake.jlama.tensor.AbstractTensor;

import java.util.Map;

public interface WeightLoader extends AutoCloseable {

    Map<String, String> metadata();

    Map<String, TensorInfo> tensorInfoMap();

    default boolean isWeightPresent(String name) {
        return tensorInfoMap().containsKey(name);
    }

    default AbstractTensor load(String name) {
        return load(name, null, false, false);
    }

    AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns);

    DType getModelDType();
}
