package com.github.tjake.jlama.tensor.operations;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.Function;

import com.github.tjake.jlama.util.MachineSpec;
import com.github.tjake.jlama.util.RuntimeSupport;
import jdk.incubator.vector.FloatVector;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.BFloat16BufferTensor;
import com.github.tjake.jlama.tensor.Float16BufferTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;

import static com.github.tjake.jlama.tensor.operations.NativeTensorOperations.*;

public class TestOperations
{
    private static final Random r = new Random(1337);
    private static final Logger logger = LoggerFactory.getLogger(TensorOperations.class);
    private static final int SIZE = 1024;
    private static final List<TensorOperations> opTypes = new ArrayList<>();

    private static final Map<DType, Function<AbstractTensor, AbstractTensor>> aTypes = new TreeMap<>();

    private static final Map<DType, Function<AbstractTensor, AbstractTensor>> bTypes = new TreeMap<>();

    private static final NaiveTensorOperations controlOps = new NaiveTensorOperations();

    @BeforeClass
    public static void init() {
        opTypes.add(new NaiveTensorOperations());
        opTypes.add(new PanamaTensorOperations());
        /*opTypes.add(new NativeTensorOperations());
        opTypes.add(new NativeTensorOperations(0));

        if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.AVX_512)
            opTypes.add(new NativeTensorOperations(HAS_AVX2));

        if (RuntimeSupport.isLinux() || RuntimeSupport.isWin()) {
            opTypes.add(new NativeTensorOperations(HAS_F16C));
            if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.AVX_512)
                opTypes.add(new NativeTensorOperations(HAS_F16C | HAS_AVX2));
        }*/

        aTypes.put(DType.F32, FloatBufferTensor::new);
        aTypes.put(DType.F16, Float16BufferTensor::new);
        aTypes.put(DType.BF16, BFloat16BufferTensor::new);

        bTypes.put(DType.F32, FloatBufferTensor::new);
        bTypes.put(DType.F16, Float16BufferTensor::new);
        bTypes.put(DType.BF16, BFloat16BufferTensor::new);
        bTypes.put(DType.I8, Q8ByteBufferTensor::new);
        bTypes.put(DType.Q4, Q4ByteBufferTensor::new);
    }

    static AbstractTensor makeTensor(int size) {
        FloatBufferTensor f = new FloatBufferTensor(size);
        for (int i = 0; i < size; i++)
            f.set(r.nextFloat(), i);

        return f;
    }

    @Test
    public void testDotProduct() {
        AbstractTensor a = makeTensor(SIZE);
        AbstractTensor b = makeTensor(SIZE);

        //This is what we compare others to
        float control = controlOps.dotProduct(a, b, SIZE);
        int opt = 0;
        for (TensorOperations t : opTypes) {
            int supported = 0;

            for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> aType : aTypes.entrySet()) {
                for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> bType : bTypes.entrySet()) {
                    try {
                        logger.debug("Running {} {} {}", opt, aType, bType);
                        float dp = t.dotProduct(aType.getValue().apply(a), bType.getValue().apply(b), SIZE);
                        supported++;
                        Assert.assertEquals("OP " + opt + ", AType " + aType.getKey() + ", BType " + bType.getKey() + " combo is outside of 1% error limit", control, dp, control * .01f);
                    } catch (UnsupportedOperationException | IllegalArgumentException e) {
                        logger.debug("No support for AType {} and BType {}", aType.getKey(), bType.getKey());
                    }
                }
            }
            Assert.assertTrue(supported > 0);
            opt++;
        }
    }

    @Test
    public void testAccumulate() {
        AbstractTensor a = makeTensor(SIZE);
        AbstractTensor b = makeTensor(SIZE);

        // Make a copy for testing before its changed by operation
        AbstractTensor c = new FloatBufferTensor(a);

        //This is what we compare others to
        controlOps.accumulate(a, b);
        float control = controlOps.sum(a);
        for (TensorOperations t : opTypes) {
            int supported = 0;

            for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> aType : aTypes.entrySet()) {
                for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> bType : bTypes.entrySet()) {
                    AbstractTensor chat = aType.getValue().apply(c);
                    AbstractTensor bhat = bType.getValue().apply(b);

                    try {
                        t.accumulate(chat, bhat);
                        float dp = t.sum(chat);
                        supported++;
                        Assert.assertEquals("AType " + aType.getKey() + ", BType " + bType.getKey() + " combo is outside of 1% error limit", control, dp, control * .01f);
                    } catch (UnsupportedOperationException | IllegalArgumentException e) {
                        logger.debug("No support for AType {} and BType {}", aType.getKey(), bType.getKey());
                    }
                }
            }
            Assert.assertTrue(supported > 0);
        }
    }

    @Test
    public void testScale() {
        AbstractTensor a = makeTensor(SIZE);

        // Make a copy for testing before its changed by operation
        AbstractTensor c = new FloatBufferTensor(a);

        //This is what we compare others to
        controlOps.scale(3.14159f, a, 0, SIZE);
        float control = controlOps.sum(a);

        for (TensorOperations t : opTypes) {
            int supported = 0;
            for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> aType : aTypes.entrySet()) {
                AbstractTensor chat = aType.getValue().apply(c);
                try {
                    t.scale(3.14159f, chat, 0, SIZE);
                    float dp = t.sum(chat);
                    supported++;
                    Assert.assertEquals("AType " + aType.getKey() + " is outside of 1% error limit", control, dp, control * .01f);
                } catch (UnsupportedOperationException | IllegalArgumentException e) {
                    logger.debug("No support for AType {}", aType.getKey());
                }
            }
            Assert.assertTrue(supported > 0);
        }
    }

    @Test
    public void testSaxpy() {
        AbstractTensor a = makeTensor(SIZE);
        AbstractTensor b = makeTensor(SIZE);

        // Make a copy for testing before its changed by operation
        AbstractTensor c = new FloatBufferTensor(b);

        //This is what we compare others to
        controlOps.saxpy(3.14159f, a, b, 0, 0, SIZE);
        float control = controlOps.sum(b);

        for (TensorOperations t : opTypes) {
            int supported = 0;

            for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> aType : aTypes.entrySet()) {
                for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> bType : bTypes.entrySet()) {
                    AbstractTensor ahat = bType.getValue().apply(a);
                    AbstractTensor chat = aType.getValue().apply(c);

                    try {
                        t.saxpy(3.14159f, ahat, chat, 0, 0, SIZE);
                        float dp = t.sum(chat);
                        supported++;
                        Assert.assertEquals("AType " + aType.getKey() + ", BType " + bType.getKey() + " combo is outside of 1% error limit", control, dp, control * .01f);
                    } catch (UnsupportedOperationException | IllegalArgumentException e) {
                        logger.debug("No support for AType {} and BType {}", aType.getKey(), bType.getKey());
                    }
                }
            }
            Assert.assertTrue(supported > 0);
        }
    }

    @Test
    public void testSxpby() {
        AbstractTensor a = makeTensor(SIZE);
        AbstractTensor b = makeTensor(SIZE);

        // Make a copy for testing before its changed by operation
        AbstractTensor c = new FloatBufferTensor(b);

        //This is what we compare others to
        controlOps.sxpby(3.14159f, a, b, 0, 0, SIZE);
        float control = controlOps.sum(b);

        for (TensorOperations t : opTypes) {
            int supported = 0;

            for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> aType : aTypes.entrySet()) {
                for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> bType : bTypes.entrySet()) {
                    AbstractTensor ahat = bType.getValue().apply(a);
                    AbstractTensor chat = aType.getValue().apply(c);

                    try {
                        t.sxpby(3.14159f, ahat, chat, 0, 0, SIZE);
                        float dp = t.sum(chat);
                        supported++;
                        Assert.assertEquals("AType " + aType.getKey() + ", BType " + bType.getKey() + " combo is outside of 1% error limit", control, dp, control * .01f);
                    } catch (UnsupportedOperationException | IllegalArgumentException e) {
                        logger.debug("No support for AType {} and BType {}", aType.getKey(), bType.getKey());
                    }
                }
            }
            Assert.assertTrue(supported > 0);
        }
    }
}
