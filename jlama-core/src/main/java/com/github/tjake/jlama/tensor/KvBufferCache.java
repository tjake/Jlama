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
package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.DistributedContext;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.util.Pair;
import java.io.IOError;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * A cache for key-value buffers used in the model.
 * @see com.github.tjake.jlama.model.functions.Generator
 */
public class KvBufferCache {

    public static final String TOKEN_COUNT = "TOKEN_COUNT";

    private final ConcurrentMap<UUID, Pair<RandomAccessFile, AbstractTensor>> kvBufferCache;
    private final AbstractModel model;

    public KvBufferCache(AbstractModel model) {
        this.kvBufferCache = new ConcurrentHashMap<>();
        this.model = model;
    }

    public AbstractTensor getKvBuffer(UUID session) {
        return kvBufferCache.computeIfAbsent(session, this::makeKvBuffer).right;
    }

    private Pair<RandomAccessFile, AbstractTensor> makeKvBuffer(UUID session) {
        TensorShape s;
        Config c = model.getConfig();
        DistributedContext dctx = c.dctx();
        // FIXME: Max size should be configurable
        int[] rawShape = new int[] { dctx.numberOfLayers, 2, Math.min(1024, c.contextLength), c.kvLength };

        // Adjust the shape to be relative to the kv cache size (in case of GQA)
        if (c.kvLength != dctx.kvSegmentLength) {
            Pair<Integer, Integer> kvOffset = Pair.of(dctx.kvSegmentStart, dctx.kvSegmentEnd);
            s = TensorShape.sparseColumn(rawShape, kvOffset);
        } else {
            s = TensorShape.of(rawShape);
        }

        // If we don't have a working directory, just use a FloatBufferTensor
        if (model.getConfig().workingDirectory().isEmpty()) {
            return Pair.of(null, AbstractTensor.make(model.getWorkingDType(), s));
        }

        // Otherwise, create a file-backed tensor
        try {
            RandomAccessFile raf = new RandomAccessFile(
                Paths.get(model.getConfig().workingDirectory().get().toString(), session.toString()).toFile(),
                "rw"
            );
            long bytes = s.size() * model.getWorkingDType().size();
            raf.setLength(bytes);

            AbstractTensor t;
            if (model.getWorkingDType() == DType.F32) {
                FloatBuffer fb = raf.getChannel()
                    .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer();

                t = new FloatBufferTensor(fb, s, true);
            } else if (model.getWorkingDType() == DType.BF16) {
                ShortBuffer sb = raf.getChannel()
                    .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asShortBuffer();

                t = new BFloat16BufferTensor("kvmem", sb, s, true);
            } else {
                throw new UnsupportedOperationException("Only F32/BF16 is supported for now");
            }

            return Pair.of(raf, t);

        } catch (IOException e) {
            throw new IOError(e);
        }
    }
}
