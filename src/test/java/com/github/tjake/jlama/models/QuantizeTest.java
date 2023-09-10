package com.github.tjake.jlama.models;

import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.*;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class QuantizeTest {

    @Test
    public void testQuantizeF16Quantize() {
        short f1 = FloatConversions.float32ToBFloat16(0.1f);
        float f3 = FloatConversions.bFloat16ToFloat32(f1);

        Assert.assertEquals( 0.1f, f3, 0.0001f);
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
    public void testF32Q8DotProd() {
        FloatBufferTensor f1 = new FloatBufferTensor(1024);
        FloatBufferTensor f2 = new FloatBufferTensor(1024);
        for (int i = 0; i < 1024; i++) {
            f1.set(ThreadLocalRandom.current().nextFloat(), i);
            f2.set(ThreadLocalRandom.current().nextFloat(), i);
        }

        Q8ByteBufferTensor b1 = new Q8ByteBufferTensor(f1);
        Q8ByteBufferTensor b2 = new Q8ByteBufferTensor(f2);

        for (int i = 0; i < 1024; i++) {
            Assert.assertEquals(f1.get(i), b1.get(i), 0.01f);
            Assert.assertEquals(f2.get(i), b2.get(i), 0.01f);
        }

        float dot32 = VectorMath.dotProduct(f1, f2, 1024);
        float dot8  = VectorMath.dotProduct(b1, b2, 1024);
        float dot32_8  = VectorMath.dotProduct(f1, b2, 1024);

        Assert.assertEquals(dot32, dot8, 1f);
        Assert.assertEquals(dot32, dot32_8, 1f);
    }

    @Test
    public void testQ8DotProd() {
        FloatBufferTensor f1 = new FloatBufferTensor(1024);
        FloatBufferTensor f2 = new FloatBufferTensor(1024);
        for (int i = 0; i < 1024; i++) {
            f1.set(ThreadLocalRandom.current().nextFloat(), i);
            f2.set(ThreadLocalRandom.current().nextFloat(), i);
        }

        Q8ByteBufferTensor b1 = new Q8ByteBufferTensor(f1);
        Q8ByteBufferTensor b2 = new Q8ByteBufferTensor(f2);

        for (int i = 0; i < 1024; i++) {
            Assert.assertEquals(f1.get(i), b1.get(i), 0.01f);
            Assert.assertEquals(f2.get(i), b2.get(i), 0.01f);
        }

        float dot32 = VectorMath.dotProduct(f1, f2, 1024);
        float dot8  = VectorMath.dotProduct(b1, b2, 1024);
        Assert.assertEquals(dot32, dot8, 1f);
    }

    @Test
    public void testQ5() {
        Random r = new Random(1337);

        FloatBufferTensor f1 = new FloatBufferTensor(1024);
        FloatBufferTensor f2 = new FloatBufferTensor(1024);
        for (int i = 0; i < 1024; i++) {
            f1.set(r.nextFloat(), i);
            f2.set(r.nextFloat(), i);
        }

        Q5ByteBufferTensor b1 = new Q5ByteBufferTensor(f1);
        Q5ByteBufferTensor b2 = new Q5ByteBufferTensor(f2);

        for (int i = 0; i < 1024; i++) {
            Assert.assertEquals("i=" + i, f1.get(i), b1.get(i), 0.05f);
            Assert.assertEquals("i=" + i, f2.get(i), b2.get(i), 0.05f);
        }
    }

    @Test
    public void testQ4() {
        Random r = new Random(1337);
        FloatBufferTensor f1 = new FloatBufferTensor(1024);
        FloatBufferTensor f2 = new FloatBufferTensor(1024);
        for (int i = 0; i < 1024; i++) {
            f1.set(r.nextFloat(), i);
            f2.set(r.nextFloat(), i);
        }

        Q4ByteBufferTensor b1 = new Q4ByteBufferTensor(f1);
        Q4ByteBufferTensor b2 = new Q4ByteBufferTensor(f2);

        for (int i = 0; i < 1024; i++) {
            Assert.assertEquals("i=" + i, f1.get(i), b1.get(i), 0.5f);
            Assert.assertEquals("i=" + i, f2.get(i), b2.get(i), 0.5f);
        }
    }


    @Test
    public void testQ4DotProd() {
        Random r = new Random(1337);
        FloatBufferTensor f1 = new FloatBufferTensor(1024);
        FloatBufferTensor f2 = new FloatBufferTensor(1024);
        for (int i = 0; i < 1024; i++) {
            f1.set(r.nextFloat(), i);
            f2.set(r.nextFloat(), i);
        }

        Q4ByteBufferTensor b2 = new Q4ByteBufferTensor(f2);
        for (int i = 0; i < 1024; i++) {
            Assert.assertEquals(f2.get(i), b2.get(i), 0.1f);
        }

        float dot32 = VectorMath.dotProduct(f1, f2, 1024);

        float mdot4 = 0f;
        for (int i = 0; i < 1024; i++) {
            mdot4 += f1.get(i) * b2.get(i);
        }
        Assert.assertEquals(dot32, mdot4, 1f);

        float dot4  = VectorMath.dotProduct(f1, b2, 1024);
        Assert.assertEquals(dot32, dot4, 1f);

        Assert.assertEquals(mdot4, dot4, .1f);
    }
}
