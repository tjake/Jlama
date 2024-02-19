package com.github.tjake.jlama.util;

import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteOrder;

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
            if (preferredBits == 512)
                tmp = Type.AVX_512;

            if (preferredBits == 256)
                tmp = Type.AVX_256;

            if (preferredBits == 128 && RuntimeSupport.isArm())
                tmp = Type.ARM_128;

        } catch (Throwable t) {
            logger.warn("Java SIMD Vector API *not* available. Add --add-modules=jdk.incubator.vector to your JVM options");
        }

        logger.debug("Machine Vector Spec: {}", tmp);
        logger.debug("Byte Order: {}", ByteOrder.nativeOrder().toString());
        type = tmp;
    }

}
