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
package com.github.tjake.jlama.util;

import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Detects the machine's vector capabilities
 */
public class MachineSpec {
    private static final Logger logger = LoggerFactory.getLogger(MachineSpec.class);
    public static final Type VECTOR_TYPE = new MachineSpec().type;

    public enum Type {
        AVX_256(2),
        AVX_512(4),
        ARM_128(8),
        NONE(0);

        public final int ctag;

        Type(int cflag) {
            this.ctag = cflag;
        }
    }

    private final Type type;

    private MachineSpec() {
        Type tmp = Type.NONE;
        try {
            int preferredBits = FloatVector.SPECIES_PREFERRED.vectorBitSize();
            if (preferredBits == 512) tmp = Type.AVX_512;

            if (preferredBits == 256) tmp = Type.AVX_256;

            if (preferredBits == 128 && RuntimeSupport.isArm()) tmp = Type.ARM_128;

            if (tmp == Type.NONE) logger.warn("Unknown vector type: {}", preferredBits);

        } catch (Throwable t) {
            logger.warn("Java SIMD Vector API *not* available. Add --add-modules=jdk.incubator.vector to your JVM options");
        }

        logger.debug("Machine Vector Spec: {}", tmp);
        logger.debug("Byte Order: {}", ByteOrder.nativeOrder().toString());
        type = tmp;
    }
}
