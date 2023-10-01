package com.github.tjake.jlama.tensor.operations;


import com.github.tjake.jlama.util.MachineSpec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.tjake.jlama.tensor.operations.cnative.NativeSimd;
import jdk.incubator.vector.VectorOperators;

public class TensorOperationsProvider
{
    private static final Logger logger = LoggerFactory.getLogger(TensorOperationsProvider.class);

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
           // NativeSimd.accumulate_f16$MH();
            //return new NativeTensorOperations();
        } catch (Throwable t) {
            logger.info("Error loading native operations", t);
        }

        return MachineSpec.VECTOR_TYPE == MachineSpec.Type.NONE ? new NaiveTensorOperations() : new PanamaTensorOperations();
    }
}
