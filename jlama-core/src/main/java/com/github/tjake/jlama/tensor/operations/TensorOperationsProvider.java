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
package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.util.MachineSpec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TensorOperationsProvider {
    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    private static final Logger logger = LoggerFactory.getLogger(TensorOperationsProvider.class);
    private static final boolean forcePanama = Boolean.getBoolean("jlama.force_panama_tensor_operations");

    private static final String lock = "lock";
    private static TensorOperationsProvider instance;

    public static TensorOperations get() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) instance = new TensorOperationsProvider();
            }
        }

        return instance.provider;
    }

    private final TensorOperations provider;

    private TensorOperationsProvider() {
        this.provider = pickFastestImplementation();
    }

    private TensorOperations pickFastestImplementation() {

        TensorOperations pick = null;

        if (!forcePanama) {
            try {
                Class<? extends TensorOperations> nativeClazz = (Class<? extends TensorOperations>) Class.forName(
                    "com.github.tjake.jlama.tensor.operations.NativeSimdTensorOperations"
                );
                pick = nativeClazz.getConstructor().newInstance();
                // This will throw if no shared lib found
            } catch (Throwable t) {
                logger.warn("Native operations not available. Consider adding 'com.github.tjake:jlama-native' to the classpath");
            }
        }

        if (pick == null) pick = MachineSpec.VECTOR_TYPE == MachineSpec.Type.NONE
            ? new NaiveTensorOperations()
            : new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);

        logger.info("Using {} ({})", pick.name(), "OffHeap");
        return pick;
    }
}
