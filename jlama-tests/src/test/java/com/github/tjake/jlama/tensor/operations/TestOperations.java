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

import static com.github.tjake.jlama.tensor.operations.NativeTensorOperations.*;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.BFloat16BufferTensor;
import com.github.tjake.jlama.tensor.Float16BufferTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.util.MachineSpec;
import com.github.tjake.jlama.util.RuntimeSupport;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.Function;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestOperations {
    private static final Random r = new Random();
    private static final Logger logger = LoggerFactory.getLogger(TensorOperations.class);
    private static final int SIZE = 1024;
    private static final int ROWS = 128;
    private static final List<TensorOperations> opTypes = new ArrayList<>();

    private static final Map<DType, Function<AbstractTensor, AbstractTensor>> aTypes = new TreeMap<>();

    private static final Map<DType, Function<AbstractTensor, AbstractTensor>> bTypes = new TreeMap<>();

    private static final NaiveTensorOperations controlOps = new NaiveTensorOperations();
    private static final PanamaTensorOperations vectorOps = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
    private static final TensorOperations globalOps = TensorOperationsProvider.get();

    @BeforeClass
    public static void init() {
        logger.info("Globally using {}", globalOps.name());
        opTypes.add(new NaiveTensorOperations());
        opTypes.add(new PanamaTensorOperations(MachineSpec.Type.AVX_512));
        opTypes.add(new PanamaTensorOperations(MachineSpec.Type.AVX_256));
        opTypes.add(new PanamaTensorOperations(MachineSpec.Type.ARM_128));

        if (globalOps instanceof NativeTensorOperations) {
            opTypes.add(new NativeTensorOperations());
            opTypes.add(new NativeTensorOperations(0));

            if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.AVX_512) opTypes.add(new NativeTensorOperations(HAS_AVX2));

            if (RuntimeSupport.isLinux() || RuntimeSupport.isWin()) {
                opTypes.add(new NativeTensorOperations(HAS_F16C));
                if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.AVX_512)
                    opTypes.add(new NativeTensorOperations(HAS_F16C | HAS_AVX2));
            }

            if (RuntimeSupport.isArm()) {
                opTypes.add(new NativeTensorOperations(MachineSpec.Type.ARM_128.ctag));
            }
        }

        aTypes.put(DType.F32, FloatBufferTensor::new);
        aTypes.put(DType.F16, Float16BufferTensor::new);
        aTypes.put(DType.BF16, BFloat16BufferTensor::new);
        aTypes.put(DType.I8, Q8ByteBufferTensor::new);

        bTypes.put(DType.F32, FloatBufferTensor::new);
        bTypes.put(DType.F16, Float16BufferTensor::new);
        bTypes.put(DType.BF16, BFloat16BufferTensor::new);
        bTypes.put(DType.I8, Q8ByteBufferTensor::new);
        bTypes.put(DType.Q4, Q4ByteBufferTensor::new);
    }

    static FloatBufferTensor makeTensor(int size) {
        FloatBufferTensor f = new FloatBufferTensor(size);
        for (int i = 0; i < size; i++) f.set(r.nextFloat(), i);

        return f;
    }

    static FloatBufferTensor makeWeights(int rows, int size) {
        FloatBufferTensor f = new FloatBufferTensor(rows, size);
        for (int j = 0; j < rows; j++) for (int i = 0; i < size; i++) f.set(r.nextFloat(), j, i);

        return f;
    }

    @Test
    public void testDotProduct() {
        AbstractTensor a = makeTensor(SIZE);
        AbstractTensor b = makeTensor(SIZE);

        // This is what we compare others to
        float control = controlOps.dotProduct(a, b, SIZE);
        int opt = 0;
        for (TensorOperations t : opTypes) {
            int supported = 0;

            for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> aType : aTypes.entrySet()) {
                for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> bType : bTypes.entrySet()) {
                    try {
                        logger.debug("Running {} {} {}", opt, aType, bType);
                        float dp = t.dotProduct(
                                aType.getValue().apply(a), bType.getValue().apply(b), SIZE);
                        supported++;
                        Assert.assertEquals(
                                "OP " + t.name() + ", AType " + aType.getKey() + ", BType " + bType.getKey()
                                        + " combo is outside of 1% error limit",
                                control,
                                dp,
                                control * .01f);
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
    public void testSplitDotProduct() {
        AbstractTensor a = makeTensor(SIZE);
        AbstractTensor b = makeTensor(SIZE);

        // This is what we compare others to
        float control = controlOps.dotProduct(a, b, SIZE);

        float p1 = controlOps.dotProduct(a, b, 0, 0, SIZE / 2);
        float p2 = controlOps.dotProduct(a, b, SIZE / 2, SIZE / 2, SIZE / 2);

        Assert.assertEquals(control, p1 + p2, control * .01f);
    }

    @Test
    public void testAccumulate() {
        AbstractTensor a = makeTensor(SIZE);
        AbstractTensor b = makeTensor(SIZE);

        // Make a copy for testing before its changed by operation
        AbstractTensor c = new FloatBufferTensor(a);

        // This is what we compare others to
        controlOps.accumulate(a, b, 0, SIZE);
        float control = controlOps.sum(a);
        for (TensorOperations t : opTypes) {
            int supported = 0;
            for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> aType : aTypes.entrySet()) {
                for (Map.Entry<DType, Function<AbstractTensor, AbstractTensor>> bType : bTypes.entrySet()) {
                    AbstractTensor chat = aType.getValue().apply(c);
                    AbstractTensor bhat = bType.getValue().apply(b);

                    try {
                        t.accumulate(chat, bhat, 0, SIZE);
                        float dp = t.sum(chat);
                        supported++;
                        Assert.assertEquals(
                                "AType " + aType.getKey() + ", BType " + bType.getKey()
                                        + " combo is outside of 1% error limit",
                                control,
                                dp,
                                control * .01f);
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

        // This is what we compare others to
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
                    Assert.assertEquals(
                            "AType " + aType.getKey() + " is outside of 1% error limit", control, dp, control * .01f);
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

        // This is what we compare others to
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
                        Assert.assertEquals(
                                "AType " + aType.getKey() + ", BType " + bType.getKey()
                                        + " combo is outside of 1% error limit",
                                control,
                                dp,
                                control * .01f);
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

        // This is what we compare others to
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
                        Assert.assertEquals(
                                "AType " + aType.getKey() + ", BType " + bType.getKey()
                                        + " combo is outside of 1% error limit",
                                control,
                                dp,
                                control * .01f);
                    } catch (UnsupportedOperationException | IllegalArgumentException e) {
                        logger.debug("No support for AType {} and BType {}", aType.getKey(), bType.getKey());
                    }
                }
            }
            Assert.assertTrue(supported > 0);
        }
    }

    @Test
    public void testQ8Vectorized() {
        FloatBufferTensor a = makeTensor(SIZE);

        Q8ByteBufferTensor ref = new Q8ByteBufferTensor(a);
        Q8ByteBufferTensor qv = vectorOps.quantizeQ8_256(a, 0, (int) a.size());
        Q8ByteBufferTensor qv1 = vectorOps.quantizeQ8_512(a, 0, (int) a.size());
        Q8ByteBufferTensor qv2 = vectorOps.quantizeQ8_arm(a, 0, (int) a.size());

        Assert.assertEquals(controlOps.sum(ref), controlOps.sum(qv), 0.0001);
        Assert.assertEquals(controlOps.sum(ref), controlOps.sum(qv1), 0.0001);
        Assert.assertEquals(controlOps.sum(ref), controlOps.sum(qv2), 0.0001);
    }

    @Test
    public void testBatchChunked() {
        FloatBufferTensor r0 = makeTensor(ROWS);
        FloatBufferTensor r1 = makeTensor(ROWS);

        FloatBufferTensor b0 = makeTensor(ROWS);
        FloatBufferTensor b1 = makeTensor(ROWS);

        FloatBufferTensor a = makeTensor(SIZE);

        FloatBufferTensor w0 = makeWeights(ROWS, SIZE);
        FloatBufferTensor w1 = makeWeights(ROWS, SIZE);

        controlOps.dotProductChunk(r0, a, w0, 0, SIZE, 0, ROWS);
        controlOps.dotProductChunk(r1, a, w1, 0, SIZE, 0, ROWS);

        for (TensorOperations t : opTypes) {
            t.dotProductBatchChunk(new AbstractTensor[] {b0, b1}, a, new AbstractTensor[] {w0, w1}, 0, SIZE, 0, ROWS);

            Assert.assertEquals(controlOps.sum(r0), controlOps.sum(b0), 0.01);
            Assert.assertEquals(controlOps.sum(r1), controlOps.sum(b1), 0.01);
        }
    }
}
