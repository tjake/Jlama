package com.github.tjake.jlama.models;

import com.github.tjake.jlama.model.FloatBufferTensor;
import com.github.tjake.jlama.model.Tensor;
import org.junit.Test;

import java.util.concurrent.ThreadLocalRandom;

public class QuantizeTest {

    @Test
    public void testQuantizeF16Quantize() {

    }

    Tensor createVector(int size) {
        Tensor t = new FloatBufferTensor(size);
        for (int i = 0; i < size; i++) {
            t.set(ThreadLocalRandom.current().nextFloat(), i);
        }
        return t;
    }

}
