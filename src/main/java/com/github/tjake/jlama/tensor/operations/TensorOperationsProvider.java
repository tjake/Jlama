package com.github.tjake.jlama.tensor.operations;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.tjake.jlama.tensor.operations.cnative.NativeSimd;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

public class TensorOperationsProvider
{
    private static final Logger logger = LoggerFactory.getLogger(TensorOperationsProvider.class);
    private static boolean hasVectorAPI = hasVectorAPI();
    static final boolean hasAVX2 = FloatVector.SPECIES_PREFERRED == FloatVector.SPECIES_512;

    private static boolean hasVectorAPI() {
        try {
            VectorOperators.ADD.name();
            logger.info("Java 20+ Vector API available");
            if (hasAVX2)
                logger.info("AVX2 operations available");
            return true;
        } catch (Throwable t) {
            logger.warn("Java SIMD Vector API *not* available. Missing --add-modules=jdk.incubator.vector?");
            return false;
        }
    }
    private static final String lock = "lock";
    private static TensorOperationsProvider instance;
    public static TensorOperations get() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null)
                    instance = new TensorOperationsProvider();
            }
        }

        return instance.provider;
    }

    private final TensorOperations provider;
    private TensorOperationsProvider() {
        this.provider = pickFastestImplementaion();
    }

    private TensorOperations pickFastestImplementaion() {
        try {
            NativeSimd.accumulate_f16$MH();
            return new NativeTensorOperations();
        } catch (Throwable t) {
            logger.info("Error loading native operations", t);
        }
        return hasVectorAPI ? new PanamaTensorOperations() : new NaiveTensorOperations();
    }
}
