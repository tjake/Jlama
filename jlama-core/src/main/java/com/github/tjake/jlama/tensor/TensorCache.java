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

import com.github.tjake.jlama.safetensors.DType;
import com.google.common.collect.Maps;
import java.util.Objects;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import org.jctools.queues.MpmcUnboundedXaddArrayQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * In LLMs a lot of buffers are used for inference.  Rather than allocating each one or using a fixed pool
 * this TensorCache allows a limited number of different shaped buffers to be reused across threads
 */
public class TensorCache {

    public static final TensorCache instance = new TensorCache(100 * 1024 * 1024);

    private static final Logger logger = LoggerFactory.getLogger(TensorCache.class);

    public static class ShapeKey {
        final TensorShape shape;
        final DType dType;

        ShapeKey(DType dType, TensorShape shape) {
            this.dType = dType;
            this.shape = shape;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ShapeKey shapeKey = (ShapeKey) o;
            return Objects.equals(shape, shapeKey.shape) && dType == shapeKey.dType;
        }

        @Override
        public int hashCode() {
            return Objects.hash(shape, dType);
        }
    }

    private final long bytesCapacity;
    private final AtomicLong currentBytes;
    private final ConcurrentMap<ShapeKey, MpmcUnboundedXaddArrayQueue<AbstractTensor>> availableByShape;

    private final Function<ShapeKey, MpmcUnboundedXaddArrayQueue<AbstractTensor>> queueFactory = s -> new MpmcUnboundedXaddArrayQueue<>(
        128
    );

    public TensorCache(long bytesCapacity) {
        this.bytesCapacity = bytesCapacity;
        this.currentBytes = new AtomicLong(0);
        this.availableByShape = Maps.newConcurrentMap();
    }

    public AbstractTensor get(DType dType, TensorShape shape) {
        MpmcUnboundedXaddArrayQueue<AbstractTensor> availableQueue = availableByShape.computeIfAbsent(
            new ShapeKey(dType, shape),
            queueFactory
        );
        AbstractTensor t = availableQueue.poll();

        if (t != null) return t;

        t = switch (dType) {
            case F32 -> new FloatBufferTensor(shape);
            case F16 -> new Float16BufferTensor(shape);
            case BF16 -> new BFloat16BufferTensor(shape);
            case I8 -> new Q8ByteBufferTensor(shape);
            case Q4 -> new Q4ByteBufferTensor(shape);
            default -> throw new RuntimeException("Unsupported tensor type: " + dType);
        };

        // Assign to this cache or just over allocate
        if (currentBytes.addAndGet(t.size()) < bytesCapacity) {
            t.setOwnerCache(this);
        } else {
            logger.debug("Full!");
            currentBytes.addAndGet(-t.size());
        }

        return t;
    }

    void release(AbstractTensor b) {
        b.clear();
        MpmcUnboundedXaddArrayQueue<AbstractTensor> availableQueue = availableByShape.computeIfAbsent(
            new ShapeKey(b.dType(), b.shape()),
            queueFactory
        );
        availableQueue.offer(b);
    }
}
