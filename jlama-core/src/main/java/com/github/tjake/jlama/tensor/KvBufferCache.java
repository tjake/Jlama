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
import com.google.common.base.Preconditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOError;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A cache for key-value buffers used in the model.
 * @see com.github.tjake.jlama.model.functions.Generator
 */
public class KvBufferCache {
    private static final Logger logger = LoggerFactory.getLogger(KvBufferCache.class);
    private final ConcurrentMap<UUID, KvBuffer> kvBufferCache;
    private final AbstractModel model;

    public KvBufferCache(AbstractModel model) {
        this.kvBufferCache = new ConcurrentHashMap<>();
        this.model = model;
    }

    public KvBuffer getKvBuffer(UUID session) {
        return kvBufferCache.computeIfAbsent(session, s -> new KvBuffer(s, 1 << 24)); //16MB per page
    }

    class KvPageContext {
        public final int numberOfLayerPages;
        public final int numberOfContextPages;
        private final int layersPerPage;
        private final int contextLengthPerPage;
        private final UUID session;

        public final TensorShape pageShape;

        public KvPageContext(UUID session, int numberOfLayerPages, int numberOfContextPages, int layersPerPage, int contextLengthPerPage) {
            this.session = session;
            this.numberOfLayerPages = numberOfLayerPages;
            this.numberOfContextPages = numberOfContextPages;
            this.layersPerPage = layersPerPage;
            this.contextLengthPerPage = contextLengthPerPage;

            if (numberOfLayerPages < 1)
                throw new IllegalArgumentException("totalPageCount must be >= 1");

            if (numberOfContextPages < 1)
                throw new IllegalArgumentException("numberOfContextPages must be >= 1");

            if (layersPerPage < 1)
                throw new IllegalArgumentException("layersPerPage must be >= 1");

            if (contextLengthPerPage < 1)
                throw new IllegalArgumentException("contextLengthPerPage must be >= 1");

            TensorShape s;
            Config c = model.getConfig();
            DistributedContext dctx = c.dctx();
            int[] rawShape = new int[] {layersPerPage, 2, contextLengthPerPage, c.kvLength};

            // Adjust the shape to be relative to the kv cache size (in case of GQA)
            if (c.kvLength != dctx.kvSegmentLength) {
                Pair<Integer, Integer> kvOffset = Pair.of(dctx.kvSegmentStart, dctx.kvSegmentEnd);
                s = TensorShape.sparseColumn(rawShape, kvOffset);
            } else {
                s = TensorShape.of(rawShape);
            }

            this.pageShape = s;
        }
    }

    /**
     * A Page of a key-value buffer.
     * Rather than allocating one giant buffer for the entire key-value buffer, we allocate slices of the buffer
     * as needed. This allows us to keep the memory usage low, and also allows us to allocate very large contexts.
     */
    class KvBufferPage implements AutoCloseable {
        private final AbstractTensor tensor;

        private final KvPageContext pageCtx;
        private final String pageId;

        private final RandomAccessFile raf;

        KvBufferPage(KvPageContext pageCtx, String pageId) {
            this.pageCtx = pageCtx;
            this.pageId = pageId;

            if (model.getConfig().workingDirectory().isEmpty()) {
                this.raf = null;
                this.tensor = AbstractTensor.make(model.getWorkingDType(), pageCtx.pageShape);
            } else {
                try {
                    raf = new RandomAccessFile(
                            Paths.get(model.getConfig().workingDirectory().get().toString(), pageCtx.session.toString() + "-" + pageId + ".page").toFile(),
                            "rw");
                    long bytes = pageCtx.pageShape.size() * model.getWorkingDType().size();
                    raf.setLength(bytes);

                    AbstractTensor t;
                    if (model.getWorkingDType() == DType.F32) {
                        FloatBuffer fb = raf.getChannel()
                                .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .asFloatBuffer();

                        t = new FloatBufferTensor(fb, pageCtx.pageShape, true);
                    } else if (model.getWorkingDType() == DType.BF16) {
                        ShortBuffer sb = raf.getChannel()
                                .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .asShortBuffer();

                        t = new BFloat16BufferTensor("kvmem", sb, pageCtx.pageShape, true);
                    } else {
                        throw new UnsupportedOperationException("Only F32/BF16 is supported for now");
                    }

                    this.tensor = t;

                } catch (IOException e) {
                    throw new IOError(e);
                }
            }
        }

        public AbstractTensor getTensor() {
            return tensor;
        }

        @Override
        public void close() throws IOException {
            if (raf != null) {
                raf.close();
            }
        }
    }

    public class KvBuffer implements AutoCloseable {
        private UUID session;
        private final AtomicInteger currentContextPosition = new AtomicInteger(0);
        private final KvBufferPage[][] pages;

        private final KvPageContext pageContext;

        KvBuffer(UUID session, int maxPageSizeInBytes) {
            this.session = session;
            this.pageContext = computePageSize(maxPageSizeInBytes);
            this.pages = new KvBufferPage[pageContext.numberOfLayerPages][pageContext.numberOfContextPages];
        }

        public int getCurrentContextPosition() {
            return currentContextPosition.get();
        }

        public void setCurrentContextPosition(int position) {
            currentContextPosition.set(position);
        }

        public void incrementContextPosition() {
            currentContextPosition.incrementAndGet();
        }

        public KvPageContext computePageSize(long maxPageSizeInBytes) {
            Config c = model.getConfig();
            DType workingDType = model.getWorkingDType();
            long s = 2L * workingDType.size() * c.dctx().kvSegmentLength; // Size per layer per context

            Preconditions.checkArgument(maxPageSizeInBytes > s, "maxPageSizeInBytes must be greater than the size of a single layer");

            int N = c.dctx().numberOfLayers;
            int C = c.contextLength;

            int optimalLayersPerPage = 1;
            int optimalContextLengthPerPage = 1;
            long maxProduct = 0;

            // Try partitioning by layers
            for (int x = N; x >= 1; x--) {
                long y = maxPageSizeInBytes / (x * s);

                if (y >= 1 && y <= C) {
                    long product = x * y;

                    if (product > maxProduct) {
                        optimalLayersPerPage = x;
                        optimalContextLengthPerPage = (int) y;
                        maxProduct = product;
                    }
                    // Break if product starts decreasing
                    if (product < maxProduct) {
                        break;
                    }
                }
            }

            // Try partitioning by context length
            for (int y = C; y >= 1; y--) {
                long x = maxPageSizeInBytes / (y * s);

                if (x >= 1 && x <= N) {
                    long product = x * y;

                    if (product > maxProduct) {
                        optimalLayersPerPage = (int) x;
                        optimalContextLengthPerPage = y;
                        maxProduct = product;
                    }
                    if (product < maxProduct) {
                        break;
                    }
                }
            }

            // Calculate the number of pages needed
            int numberOfLayerPages = (int) Math.ceil((double) N / optimalLayersPerPage);
            int numberOfContextPages = (int) Math.ceil((double) C / optimalContextLengthPerPage);

            // Calculate the size of each page
            long pageSize = optimalLayersPerPage * optimalContextLengthPerPage * s;

            if (pageSize > maxPageSizeInBytes) {
                throw new IllegalArgumentException("Calculation error: pageSize > maxPageSizeInBytes: " + pageSize + " > " + maxPageSizeInBytes);
            }

            return new KvPageContext(session, numberOfLayerPages, numberOfContextPages, optimalLayersPerPage, optimalContextLengthPerPage);
        }

        @Override
        public void close() {

        }

        public AbstractTensor getKeyTensorForPosition(int layerIndex, int position) {
           return getTensorForPosition(layerIndex, position, 0);
        }

        public AbstractTensor getValTensorForPosition(int layerIndex, int position) {
            return getTensorForPosition(layerIndex, position, 1);
        }

        private AbstractTensor getTensorForPosition(int layerIndex, int position, int index) {
            // Calculate page indices and relative indices
            int layerPageIndex = layerIndex / pageContext.layersPerPage;
            int contextPageIndex = position / pageContext.contextLengthPerPage;
            int relativeLayerIndex = layerIndex % pageContext.layersPerPage;
            int relativeContextIndex = position % pageContext.contextLengthPerPage;

            KvBufferPage page = pages[layerPageIndex][contextPageIndex];
            if (page == null) {
                page = new KvBufferPage(pageContext, "L" + layerPageIndex + "C" + contextPageIndex);
                pages[layerPageIndex][contextPageIndex] = page;
            }

            return page.getTensor().slice(true, relativeLayerIndex, index, relativeContextIndex);
        }

        public AbstractTensor[] getKeyTensorsUptoPosition(int layerIndex, int upperBound) {
            return getTensorsUptoPosition(layerIndex, 0, upperBound);
        }

        public AbstractTensor[] getValTensorsUptoPosition(int layerIndex, int upperBound) {
            return getTensorsUptoPosition(layerIndex, 1, upperBound);
        }

        private AbstractTensor[] getTensorsUptoPosition(int layerIndex, int index, int upperBound) {
            int layerPageIndex = layerIndex / pageContext.layersPerPage;
            int contextPageIndex = upperBound / pageContext.contextLengthPerPage;
            int relativeLayerIndex = layerIndex % pageContext.layersPerPage;

            KvBufferPage[] layerPages = pages[layerPageIndex];

            AbstractTensor[] tensors = new AbstractTensor[contextPageIndex + 1];

            for (int i = 0; i <= contextPageIndex; i++) {
                KvBufferPage page = layerPages[i];
                tensors[i] = page.getTensor().slice(true, relativeLayerIndex, index);
            }

            return tensors;
        }
    }
}
