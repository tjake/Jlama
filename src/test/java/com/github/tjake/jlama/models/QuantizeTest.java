package com.github.tjake.jlama.models;

import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.math.panama.VectorNativeSimd;
import com.github.tjake.jlama.model.Float16BufferTensor;
import com.github.tjake.jlama.model.FloatBufferTensor;
import com.github.tjake.jlama.model.AbstractTensor;
import org.junit.Assert;
import org.junit.Test;

import java.lang.foreign.*;
import java.util.concurrent.ThreadLocalRandom;

public class QuantizeTest {

    @Test
    public void testQuantizeF16Quantize() {
        short f1 = FloatConversions.float32ToBFloat16(0.1f);

        float f3 = FloatConversions.bFloat16ToFloat32(f1);

        Assert.assertEquals( 0.1f, f3, 0.0f);

    }

    @Test
    public void testIeeeFloat16Math()
    {
        short a = Float.floatToFloat16(1.0f);
        short b = Float.floatToFloat16(2.0f);

        short result = FloatConversions.addIeeeFloat16(a, b);
        Assert.assertEquals(3.0f, Float.float16ToFloat(result), 0.0f);

        for (int i = 0; i < 1000; i++) {
            float f1 = ThreadLocalRandom.current().nextFloat(-1.0f, 1.0f);
            float f2 = ThreadLocalRandom.current().nextFloat(-1.0f, 1.0f);

            float f3 = f1 + f2;
            float f4 = f1 * f2;

            short s1 = Float.floatToFloat16(f1);
            short s2 = Float.floatToFloat16(f2);

            short s3 = FloatConversions.addIeeeFloat16(s1, s2);
            short s4 = FloatConversions.mulIeeeFloat16(s2, s1);

            float f5 = Float.float16ToFloat(s3);
            float f6 = Float.float16ToFloat(s4);

            Assert.assertEquals(f3, f5, 0.01f);
            Assert.assertEquals(f4, f6, 0.01f);
        }

    }

    AbstractTensor createVector(int size) {
        AbstractTensor t = new FloatBufferTensor(size);
        for (int i = 0; i < size; i++) {
            t.set(ThreadLocalRandom.current().nextFloat(), i);
        }
        return t;
    }


    @Test
    public void testPanama() {

        float[] a = new float[]{1.1f, 2.2f, 3.3f, 7.7f};
        float[] b = new float[]{4.4f, 5.5f, 6.6f, 8.8f};

        Float16BufferTensor a16 = new Float16BufferTensor(a.length);
        Float16BufferTensor b16 = new Float16BufferTensor(b.length);

        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
            a16.set(a[i], i);
            b16.set(b[i], i);
        }

        VectorNativeSimd.debug(a16.getMemorySegment(), a.length);

        float sum2 = VectorNativeSimd.dot_product(a16.getMemorySegment(), 0, b16.getMemorySegment(), 0, a.length);

        Assert.assertEquals(sum, sum2, 0.1f);
    }

}
