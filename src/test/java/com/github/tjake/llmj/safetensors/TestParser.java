package com.github.tjake.llmj.safetensors;

import com.github.tjake.llmj.model.FloatBufferTensor;
import com.github.tjake.llmj.model.gpt2.GPT2Tokenizer;
import com.github.tjake.llmj.model.llama.LlamaTokenizer;
import com.google.common.io.BaseEncoding;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.util.Arrays;

public class TestParser {
    private static Logger logger = LoggerFactory.getLogger(TestParser.class);

    @Test
    public void simpleTest()
    {
        byte[] preamble = BaseEncoding.base16().decode("5900000000000000");
		byte[] header = "{\"test\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"__metadata__\":{\"foo\":\"bar\"}}".getBytes();

        ByteBuffer bb = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
        bb.putFloat(1.0f);
        bb.putFloat(2.0f);
        bb.putFloat(3.0f);
        bb.putFloat(4.0f);
        byte[] data   = bb.array();

        Assert.assertEquals(16, data.length);
        byte[] serialized = new byte[preamble.length + header.length + data.length];
        System.arraycopy(preamble, 0, serialized, 0, preamble.length);
        System.arraycopy(header,0, serialized, preamble.length, header.length);
        System.arraycopy(data, 0, serialized, preamble.length + header.length, data.length);

        Weights v = SafeTensors.readBytes(ByteBuffer.wrap(serialized));
        logger.debug("model = {}", v);

        FloatBufferTensor t = v.load("test");
        logger.debug("t = {}", t);

        Assert.assertEquals(2, t.dims());
        Assert.assertEquals(1.0, t.get(0,0), 0.0001);
        Assert.assertEquals(2.0, t.get(0,1), 0.0001);
        Assert.assertEquals(3.0, t.get(1,0), 0.0001);
        Assert.assertEquals(4.0, t.get(1,1), 0.0001);

        FloatBufferTensor s1 = t.slice(0);
        logger.debug("s1 = {}", s1);

        Assert.assertEquals(1, s1.dims());
        Assert.assertEquals(1.0, s1.get(0), 0.0001);
        Assert.assertEquals(2.0, s1.get(1), 0.0001);

        FloatBufferTensor s2 = t.slice(1);
        logger.debug("s2 = {}", s2);

        Assert.assertEquals(1, s2.dims());
        Assert.assertEquals(3.0, s2.get(0), 0.0001);
        Assert.assertEquals(4.0, s2.get(1), 0.0001);

        int[] cursor = new int[t.dims()];
        int i = 0;
        do {
            logger.debug("{} => {}", Arrays.toString(cursor), t.get(cursor));
            Assert.assertTrue(i++ < 4);
        } while (t.iterate(cursor));

        FloatBufferTensor tt = t.transpose();
        int[] tcursor = new int[tt.dims()];
        i = 0;
        do {
            logger.debug("{} => {}", Arrays.toString(tcursor), tt.get(tcursor));
            Assert.assertTrue(i++ < 4);
        } while (tt.iterate(tcursor));

    }

    @Test
    public void testOffsets() {
        FloatBufferTensor b = new FloatBufferTensor(FloatBuffer.allocate(10), new int[]{50000, 768}, true);
        Assert.assertEquals(49000 * 768, b.getOffset(new int[]{49000, 0}));


        b = new FloatBufferTensor(FloatBuffer.allocate(10), new int[]{3, 7, 13}, true);

        Assert.assertEquals(0, b.getOffset(new int[]{0, 0, 0}));
        Assert.assertEquals(7*13*1, b.getOffset(new int[]{1,0,0}));
        Assert.assertEquals(7*13*2, b.getOffset(new int[]{2,0,0}));

    }

    @Test
    public void testTranspose() {
        int DIM = 768;
        FloatBufferTensor b = new FloatBufferTensor(FloatBuffer.allocate(DIM * DIM), new int[]{DIM, DIM}, true);
        int v = 0;
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) {
                v++;
                b.set((row * DIM) + col, row, col);
                Assert.assertEquals( v - 1, b.get(row, col), 1e-5f);
            }
        }

        FloatBufferTensor bt = b.transpose();
        v = 0;
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) {
                v++;
                Assert.assertEquals("col="+col+", row="+row, v - 1, bt.get(col, row), 1e-5f);
            }
        }
    }

    @Test
    public void testMMappedFile() throws IOException {
        String file = "data/gpt2-small/model.safetensors";
        try (RandomAccessFile sc = new RandomAccessFile(file, "r"))
        {
            ByteBuffer bb = sc.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, sc.length());

            Weights v = SafeTensors.readBytes(bb);

            FloatBufferTensor bias = v.load("h.8.attn.c_proj.bias");
            Assert.assertEquals(0.01406, bias.get(0), 0.00001);

            FloatBufferTensor w = v.load("wte.weight");
            Assert.assertEquals(0.00, w.get(50256, 0), 0.1);
        }
    }

    @Test
    public void testGPTTokenizer() throws IOException {
        Tokenizer tokenizer = new GPT2Tokenizer(Paths.get("data/gpt2-small/tokenizer.json"));

        String prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.";
        logger.debug("tokens = {}", tokenizer.encode(prompt));
        Assert.assertEquals(prompt, tokenizer.decode(tokenizer.encode(prompt)));
    }

    @Test
    public void testLLamaTokenizer() throws IOException {
        Tokenizer tokenizer = new LlamaTokenizer(Paths.get("data/llama2-7b-chat-hf"));

        String prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.";
        logger.debug("tokens = {}", tokenizer.encode(prompt));
        Assert.assertEquals(prompt, tokenizer.decode(tokenizer.encode(prompt)));
    }
}
