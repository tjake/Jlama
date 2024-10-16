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
package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.TensorShape;
import com.google.common.io.BaseEncoding;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestParser {
    private static Logger logger = LoggerFactory.getLogger(TestParser.class);

    @Test
    public void simpleTest() {
        byte[] preamble = BaseEncoding.base16().decode("5900000000000000");
        byte[] header = "{\"test\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"__metadata__\":{\"foo\":\"bar\"}}"
            .getBytes();

        ByteBuffer bb = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        bb.putFloat(1.0f);
        bb.putFloat(2.0f);
        bb.putFloat(3.0f);
        bb.putFloat(4.0f);
        byte[] data = bb.array();

        Assert.assertEquals(16, data.length);
        byte[] serialized = new byte[preamble.length + header.length + data.length];
        System.arraycopy(preamble, 0, serialized, 0, preamble.length);
        System.arraycopy(header, 0, serialized, preamble.length, header.length);
        System.arraycopy(data, 0, serialized, preamble.length + header.length, data.length);

        Weights v = SafeTensorSupport.readWeights(ByteBuffer.wrap(serialized));
        logger.debug("model = {}", v);

        AbstractTensor t = v.load("test");
        logger.debug("t = {}", t);

        Assert.assertEquals(2, t.dims());
        Assert.assertEquals(1.0, t.get(0, 0), 0.0001);
        Assert.assertEquals(2.0, t.get(0, 1), 0.0001);
        Assert.assertEquals(3.0, t.get(1, 0), 0.0001);
        Assert.assertEquals(4.0, t.get(1, 1), 0.0001);

        AbstractTensor s1 = t.slice(0);
        logger.debug("s1 = {}", s1);

        Assert.assertEquals(2, s1.dims());
        Assert.assertEquals(1.0, s1.get(0, 0), 0.0001);
        Assert.assertEquals(2.0, s1.get(0, 1), 0.0001);

        AbstractTensor s2 = t.slice(1);
        logger.debug("s2 = {}", s2);

        Assert.assertEquals(2, s2.dims());
        Assert.assertEquals(3.0, s2.get(0, 0), 0.0001);
        Assert.assertEquals(4.0, s2.get(0, 1), 0.0001);

        int[] cursor = new int[t.dims()];
        int i = 0;
        do {
            logger.debug("{} => {}", Arrays.toString(cursor), t.get(cursor));
            Assert.assertTrue(i++ < 4);
        } while (t.iterate(cursor));

        AbstractTensor tt = t.transpose();
        int[] tcursor = new int[tt.dims()];
        i = 0;
        do {
            logger.debug("{} => {}", Arrays.toString(tcursor), tt.get(tcursor));
            Assert.assertTrue(i++ < 4);
        } while (tt.iterate(tcursor));
    }

    @Test
    public void testOffsets() {
        FloatBufferTensor b = new FloatBufferTensor(FloatBuffer.allocate(10), TensorShape.of(50000, 768), false);
        Assert.assertEquals(49000 * 768, b.getOffset(new int[] { 49000, 0 }));

        b = new FloatBufferTensor(FloatBuffer.allocate(10), TensorShape.of(3, 7, 13), false);

        Assert.assertEquals(0, b.getOffset(new int[] { 0, 0, 0 }));
        Assert.assertEquals(7 * 13 * 1, b.getOffset(new int[] { 1, 0, 0 }));
        Assert.assertEquals(7 * 13 * 2, b.getOffset(new int[] { 2, 0, 0 }));
    }

    @Test
    public void testTranspose() {
        int DIM = 768;
        FloatBufferTensor b = new FloatBufferTensor(FloatBuffer.allocate(DIM * DIM), TensorShape.of(DIM, DIM), false);
        int v = 0;
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) {
                v++;
                b.set((row * DIM) + col, row, col);
                Assert.assertEquals(v - 1, b.get(row, col), 1e-5f);
            }
        }

        AbstractTensor bt = b.transpose();
        v = 0;
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) {
                v++;
                Assert.assertEquals("col=" + col + ", row=" + row, v - 1, bt.get(col, row), 1e-5f);
            }
        }
    }

    @Test
    public void testSparsify() {
        int DIM = 768;
        FloatBufferTensor b = new FloatBufferTensor(FloatBuffer.allocate(DIM * DIM), TensorShape.of(DIM, DIM), false);
        int v = 0;
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) {
                v++;
                b.set((row * DIM) + col, row, col);
                Assert.assertEquals(v - 1, b.get(row, col), 1e-5f);
            }
        }

        AbstractTensor bt = b.sparsify(100, 20);

        Assert.assertEquals(DIM * DIM, b.size());
        Assert.assertEquals(DIM * 20, bt.size());

        v = 0;
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) {
                v++;
                if (col >= 100 && col < 120) Assert.assertEquals("col=" + col + ", row=" + row, v - 1, bt.get(row, col), 1e-5f);
                else {
                    try {
                        bt.get(row, col);
                        Assert.fail("Should have errored trying to access value outside of sparse range");
                    } catch (Throwable t) {
                        // pass
                    }
                }
            }
        }
    }

    @Test
    public void testMMappedFile() throws IOException {
        String file = "data/gpt2/model.safetensors";
        Assume.assumeTrue(Files.exists(Paths.get(file)));
        try (RandomAccessFile sc = new RandomAccessFile(file, "r")) {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights v = SafeTensorSupport.readWeights(bb);

            AbstractTensor bias = v.load("h.8.attn.c_proj.bias");
            Assert.assertEquals(0.01406, bias.get(0), 0.00001);

            AbstractTensor w = v.load("wte.weight");
            AbstractTensor slice = w.slice(50256);
            Assert.assertEquals(-0.027689, w.get(50256, 1), 0.00001f);

            Assert.assertEquals(-0.027689, slice.get(1), 0.00001f);
        }
    }

    @Test
    public void testSegmentedTensor() throws IOException {
        String file = "../models/model-00001-of-00191.safetensors";
        Assume.assumeTrue(Files.exists(Paths.get(file)));

        SafeTensorIndex l = SafeTensorIndex.loadSingleFile(Paths.get("../models"), "model-00001-of-00191.safetensors");

        AbstractTensor t = l.load("model.embed_tokens.weight");
        TensorInfo orig = l.tensorInfoMap().get("model.embed_tokens.weight");

        Assert.assertEquals(2, t.dims());
        Assert.assertEquals(orig.shape[0], t.shape().dim(0));
        Assert.assertEquals(orig.shape[1], t.shape().dim(1));

        // Make sure we can slice the last row
        AbstractTensor s = t.slice(orig.shape[0] - 1);

    }
}
