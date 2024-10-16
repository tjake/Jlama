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
package com.github.tjake.jlama.net.grpc;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.net.*;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.protobuf.ByteString;
import com.google.protobuf.UnsafeByteOperations;
import io.grpc.stub.StreamObserver;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.*;

import jdk.incubator.vector.FloatVector;
import org.jctools.queues.MpmcArrayQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JlamaService extends JlamaServiceGrpc.JlamaServiceImplBase {
    private static final long idealBillionParamsPerWorker = Integer.getInteger("jlama.ideal_b_params", 3);

    private static final int LAYER_IDX = 0;
    private static final int HEAD_IDX = 1;

    private static final Logger logger = LoggerFactory.getLogger(JlamaService.class);
    private final AbstractModel model;
    private final int workerCount;
    private final boolean splitHeads;
    private final boolean splitLayers;
    private final int headsPerLayerShard;
    private final int numHeadShards;
    private final int layersPerShard;
    private final int numLayerShards;
    private final List<int[]> ordinalCombinations;
    private final ConcurrentMap<UUID, RegisterResponse> workers;
    private final ConcurrentMap<UUID, Runnable> discoveryActions;

    private final GeneratorGroup generatorGroup;

    private final ConcurrentMap<String, MpmcArrayQueue<Pair<CombineRequest, StreamObserver<CombineResponse>>>> combinations;

    public JlamaService(AbstractModel model, int workerCount, boolean splitHeads, boolean splitLayers) {
        Preconditions.checkArgument(
            !splitHeads || splitLayers || workerCount <= model.getConfig().numberOfKeyValueHeads,
            "Worker count must be less than or equal to number of KV heads if not splitting layers"
        );
        this.model = model;
        this.workerCount = workerCount;
        this.splitHeads = splitHeads;
        this.splitLayers = splitLayers;
        this.workers = new ConcurrentHashMap<>();
        this.discoveryActions = new ConcurrentHashMap<>();
        this.combinations = new ConcurrentHashMap<>();
        this.generatorGroup = new GeneratorGroup();
        Config c = model.getConfig();

        int tmpHeadsPerLayerShard = splitHeads ? c.numberOfKeyValueHeads / workerCount : c.numberOfKeyValueHeads;
        int tmpLayersPerShard = splitLayers ? c.numberOfLayers / workerCount : c.numberOfLayers;

        // Calculate the number of parameters per layer and use it to determine the number of heads to split per worker
        if (splitLayers && splitHeads) {
            // throw new RuntimeException("Not yet supporting splitting layers and heads together");

            long queryParams = (long) c.embeddingLength * c.embeddingLength;
            long keyValueParams = 2L * c.numberOfKeyValueHeads * c.embeddingLength * c.embeddingLength;

            // Total attention parameters with GQA
            long attentionParams = queryParams + keyValueParams;

            // Calculate the parameters for the feedforward network
            long feedforwardParams = 2L * ((long) c.embeddingLength * c.hiddenLength + (long) c.hiddenLength * c.embeddingLength);

            // Calculate the parameters for layer normalization (2 * hiddenSize for scaling and shifting)
            long layerNormParams = 2L * c.embeddingLength;

            // Parameters per transformer layer
            long paramsPerLayer = attentionParams + feedforwardParams + layerNormParams;

            // Calculate the number of heads per layer split

            // Aim for ~nB parameters per worker
            long idealParamsPerWorker = idealBillionParamsPerWorker * 1_000_000_000L;
            long paramsPerWorker = tmpLayersPerShard * paramsPerLayer;

            if (paramsPerWorker > idealParamsPerWorker) {
                tmpHeadsPerLayerShard = Math.min(
                    Math.min(workerCount, c.numberOfKeyValueHeads),
                    (int) Math.ceilDivExact(paramsPerLayer, idealParamsPerWorker)
                );
                // Round up to the nearest power of 2
                tmpHeadsPerLayerShard = nextPowerOfTwo(tmpHeadsPerLayerShard);
                tmpHeadsPerLayerShard = c.numberOfKeyValueHeads / tmpHeadsPerLayerShard;
                tmpLayersPerShard = tmpLayersPerShard * (c.numberOfKeyValueHeads / tmpHeadsPerLayerShard);
            } else {
                tmpHeadsPerLayerShard = c.numberOfKeyValueHeads;
            }
        }

        this.headsPerLayerShard = tmpHeadsPerLayerShard;
        this.numHeadShards = c.numberOfKeyValueHeads / headsPerLayerShard;
        this.layersPerShard = tmpLayersPerShard;
        this.numLayerShards = c.numberOfLayers / layersPerShard;

        logger.info("{} Layer Shards of {}, {} Head Shards of {}", numLayerShards, layersPerShard, numHeadShards, headsPerLayerShard);

        this.ordinalCombinations = new ArrayList<>(workerCount);
        for (int i = 0; i < numLayerShards; i++) {
            for (int j = 0; j < numHeadShards; j++) {
                ordinalCombinations.add(new int[] { i, j });
            }
        }
    }

    public static int nextPowerOfTwo(int n) {
        if (n <= 1) {
            return 2; // Corner case for n = 0/1
        }

        // If n is already a power of 2, return n
        if ((n & (n - 1)) == 0) {
            return n;
        }

        // Find the position of the highest set bit
        int leadingZeros = Integer.numberOfLeadingZeros(n);

        // Calculate the next power of 2
        return 1 << (32 - leadingZeros);
    }

    public void waitForReady() {
        while (true) {
            if (generatorGroup.generators.size() == workerCount) {
                generatorGroup.waitForReady();
                return;
            }
            Uninterruptibles.sleepUninterruptibly(1, TimeUnit.SECONDS);
        }
    }

    public void shutdown() {
        for (Generator g : generatorGroup.generators) {
            try {
                g.responseObserver.onCompleted();
            } catch (Exception e) {
                logger.debug("Exception when shutting down", e);
            }
        }
    }

    public ImmutableMap<UUID, RegisterResponse> getWorkers() {
        return ImmutableMap.copyOf(workers);
    }

    /**
     * Register a worker with the coordinator.  The coordinator will return the offset and length of the embedding that
     * the worker is responsible for.
     */
    @Override
    public void register(RegisterRequest request, StreamObserver<RegisterResponse> responseObserver) {
        synchronized (workers) {
            ByteBuffer bb = request.getWorkerid().asReadOnlyByteBuffer();
            UUID wid = new UUID(bb.getLong(), bb.getLong());
            if (workers.containsKey(wid)) {
                responseObserver.onNext(workers.get(wid));
                responseObserver.onCompleted();
            } else {

                if (workers.size() == workerCount) {
                    responseObserver.onError(new RuntimeException("Not accepting any more workers"));
                    return;
                }

                int workerNum = workers.size();

                RegisterResponse r = RegisterResponse.newBuilder()
                    .setHostname(request.getHostname())
                    .setPeerPort(request.getPeerPort())
                    .setModelShard(ordinalCombinations.get(workerNum)[HEAD_IDX])
                    .setNumModelShards(numHeadShards)
                    .setLayerShard(ordinalCombinations.get(workerNum)[LAYER_IDX])
                    .setNumLayerShards(numLayerShards)
                    .setWorkerOrd(workerNum)
                    .build();

                workers.put(wid, r);
                logger.info("Registered worker {} with workerNum {} of {} with {}", wid, workerNum, workerCount, r);

                responseObserver.onNext(r);
                responseObserver.onCompleted();
            }
        }
    }

    /**
     * Sends to the requesting peer the other peer it should connect to.
     * @param request
     * @param responseObserver
     */
    @Override
    public void discover(RegisterRequest request, StreamObserver<PeerInfo> responseObserver) {
        ByteBuffer bb = request.getWorkerid().asReadOnlyByteBuffer();
        UUID wid = new UUID(bb.getLong(), bb.getLong());
        // Register should have been called before this
        if (!workers.containsKey(wid)) {
            responseObserver.onError(new RuntimeException("Worker not registered"));
        } else {

            if (!splitLayers) {
                logger.error("Discover called when splitting layers = false");
                responseObserver.onError(new RuntimeException("Not splitting layers"));
            }

            Runnable action = () -> {
                RegisterResponse worker = workers.get(wid);
                int thisWorkersLayerShard = worker.getLayerShard();
                int thisWorkersHeadShard = worker.getModelShard();

                // If this is the last worker in the layer, then it should connect to the coordinator
                if (thisWorkersLayerShard == numLayerShards - 1) {
                    responseObserver.onNext(
                        PeerInfo.newBuilder()
                            .setWorkerid(request.getWorkerid())
                            .setHostname(request.getHostname())
                            .setPeerPort(request.getPeerPort())
                            .setIsCoordinator(true)
                            .build()
                    );

                    responseObserver.onCompleted();
                } else {
                    for (RegisterResponse r : workers.values()) {
                        // If this worker is the next layer shard and the same head shard, then connect to it
                        if (r.getLayerShard() == thisWorkersLayerShard + 1 && r.getModelShard() == thisWorkersHeadShard) {
                            responseObserver.onNext(
                                PeerInfo.newBuilder()
                                    .setWorkerid(r.getHostnameBytes())
                                    .setIsCoordinator(false)
                                    .setHostname(r.getHostname())
                                    .setPeerPort(r.getPeerPort())
                                    .build()
                            );

                            responseObserver.onCompleted();
                            return;
                        }
                    }

                    responseObserver.onError(new RuntimeException("No peer found"));
                }
            };

            // Once we have all the workers, then we can calculate the result and send it back to each worker that's waiting
            synchronized (discoveryActions) {
                discoveryActions.put(wid, action);

                if (workers.size() == workerCount) {
                    for (Runnable r : discoveryActions.values()) {
                        r.run();
                    }
                    discoveryActions.clear();
                }
            }
        }
    }

    public AbstractTensor generateNextOutput(UUID session, List<Integer> tokenIds, int startPosition) {
        return generatorGroup.generateNextOutput(session, tokenIds, startPosition);
    }

    public AbstractTensor generateNextOutput(UUID session, int tokenId, int position) {
        return generatorGroup.generateNextOutput(session, tokenId, position);
    }

    @Override
    public StreamObserver<GenerateRequest> generate(StreamObserver<GenerateResponse> responseObserver) {
        Generator generator = new Generator(responseObserver);
        generatorGroup.add(generator);
        logger.info("Added worker {}", generatorGroup.generators.size());
        return generator;
    }

    @Override
    public StreamObserver<CombineRequest> combine(StreamObserver<CombineResponse> responseObserver) {

        return new StreamObserver<>() {
            @Override
            public void onNext(CombineRequest request) {
                String key = String.format("%s:%d", UUID.nameUUIDFromBytes(request.getUuid().toByteArray()), request.getLayerShard());
                MpmcArrayQueue<Pair<CombineRequest, StreamObserver<CombineResponse>>> members = combinations.computeIfAbsent(
                    key,
                    k -> new MpmcArrayQueue<>(workerCount + 1)
                );
                members.add(Pair.of(request, responseObserver));
                // logger.info("GOT COMBINE REQUEST {} {}", key, members.size());
                // If we have all the workers, then we can calculate the result and send it back
                if (members.size() == numHeadShards && combinations.remove(key, members)) {
                    MemorySegment[] tensors = null;
                    for (Pair<CombineRequest, StreamObserver<CombineResponse>> f : members) {
                        if (f.left.getTensorCount() > 0) {
                            if (tensors == null) {
                                tensors = new MemorySegment[f.left.getTensorCount()];
                                for (int i = 0; i < tensors.length; i++) {
                                    ByteBuffer bb = ByteBuffer.wrap(f.left.getTensor(i).toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
                                    tensors[i] = MemorySegment.ofBuffer(bb);
                                }
                            } else {
                                for (int i = 0; i < tensors.length; i++) {
                                    MemorySegment ms = MemorySegment.ofBuffer(
                                        f.left.getTensor(i).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
                                    );
                                    // Sum float buffers
                                    accumulateF32(tensors[i], ms, (int) tensors[i].byteSize() / Float.BYTES);
                                }
                            }
                        }
                    }

                    CombineResponse.Builder responseBuilder = CombineResponse.newBuilder();

                    if (tensors != null) {
                        for (int i = 0; i < tensors.length; i++)
                            responseBuilder = responseBuilder.addTensor(
                                UnsafeByteOperations.unsafeWrap(tensors[i].asByteBuffer().order(ByteOrder.LITTLE_ENDIAN))
                            );
                    }

                    CombineResponse response = responseBuilder.build();
                    for (Pair<CombineRequest, StreamObserver<CombineResponse>> f : members) {
                        f.right.onNext(response);
                    }
                    // logger.info("Sent response to {} members", members.size());
                    members.clear();
                }
            }

            @Override
            public void onError(Throwable throwable) {}

            @Override
            public void onCompleted() {}
        };
    }

    void accumulateF32(MemorySegment a, MemorySegment b, int length) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(length);
        int i = 0;

        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            int fi = i * Float.BYTES;
            FloatVector va = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, a, fi, ByteOrder.LITTLE_ENDIAN);
            FloatVector vb = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, b, fi, ByteOrder.LITTLE_ENDIAN);
            va.add(vb).intoMemorySegment(a, fi, ByteOrder.LITTLE_ENDIAN);
        }

        // tail
        for (; i < length; i++) {
            a.set(ValueLayout.JAVA_FLOAT, i, a.get(ValueLayout.JAVA_FLOAT, i) + b.get(ValueLayout.JAVA_FLOAT, i));
        }
    }

    public class GeneratorGroup {
        private final List<Generator> generators;

        private GeneratorGroup() {
            this.generators = new ArrayList<>();
        }

        private void add(Generator generator) {
            generators.add(generator);
        }

        public void waitForReady() {
            for (Generator g : generators) {
                Uninterruptibles.awaitUninterruptibly(g.isReady());
            }
        }

        public AbstractTensor generateNextOutput(UUID session, int tokenId, int position) {
            return generateNextOutput(session, Collections.singletonList(tokenId), position);
        }

        public AbstractTensor generateNextOutput(UUID session, List<Integer> tokenIds, int startPosition) {
            Preconditions.checkArgument(generators.size() == workerCount, "Missing workers %d", workers.size());
            ByteString sid = ByteString.copyFrom(
                ByteBuffer.allocate(128).putLong(session.getMostSignificantBits()).putLong(session.getLeastSignificantBits()).flip()
            );
            GenerateResponse gr = GenerateResponse.newBuilder()
                .setSession(sid)
                .addAllTokens(tokenIds)
                .setStartPosition(startPosition)
                .build();
            for (Generator g : generators) {
                if (splitLayers) {
                    // The last layer shard sends back to coordinator from ring
                    if (g.workerAssignment.getLayerShard() == numLayerShards - 1) g.registerLatch(session);

                    // The first layer shard gets the request from the coordinator
                    if (g.workerAssignment.getLayerShard() == 0) g.responseObserver.onNext(gr);
                } else {
                    g.registerLatch(session);
                    g.responseObserver.onNext(gr);
                }
            }

            AbstractTensor output = model.makeDenseTensor(model.getConfig().embeddingLength);
            boolean found = false;
            for (int j = 0; j < workerCount; j++) {
                Generator g = generators.get(j);
                if (splitLayers && g.workerAssignment.getLayerShard() != numLayerShards - 1) continue;

                ByteString v = g.waitForOutput(session);
                output.getMemorySegment().copyFrom(MemorySegment.ofBuffer(v.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)));
                found = true;
                break;
            }

            if (!found) {
                throw new RuntimeException("No output received from workers");
            }

            // logger.info("Received output from worker {}", TensorOperationsProvider.get().sum(output));

            return output;
        }
    }

    class Generator implements StreamObserver<GenerateRequest> {
        private static final Logger logger = LoggerFactory.getLogger(Generator.class);

        private volatile UUID workerId;
        private volatile RegisterResponse workerAssignment;
        private CountDownLatch readyLatch;
        private final StreamObserver<GenerateResponse> responseObserver;
        private final ConcurrentMap<UUID, ByteString> outputs;
        private final ConcurrentMap<UUID, CountDownLatch> outputLatches;

        public Generator(StreamObserver<GenerateResponse> responseObserver) {
            this.workerId = null;
            this.workerAssignment = null;
            this.readyLatch = new CountDownLatch(1);
            this.responseObserver = responseObserver;
            this.outputs = new ConcurrentHashMap<>();
            this.outputLatches = new ConcurrentHashMap<>();
        }

        @Override
        public void onNext(GenerateRequest generateRequest) {
            if (workerId == null) {
                ByteBuffer bb = generateRequest.getWorkerid().asReadOnlyByteBuffer();
                workerId = new UUID(bb.getLong(), bb.getLong());
                workerAssignment = workers.get(workerId);
                readyLatch.countDown();
                logger.info("Worker {} ready", workerId);
                return;
            }

            ByteBuffer bb = generateRequest.getSession().asReadOnlyByteBuffer();
            UUID session = new UUID(bb.getLong(), bb.getLong());

            /*if (outputs.containsKey(session)) {
                logger.error("Previous output not consumed from worker {}", workerId);
            }*/

            outputs.put(session, generateRequest.getTensor());

            if (outputLatches.containsKey(session)) {
                outputLatches.get(session).countDown();
            } else {
                logger.error("No latch registered for session {}", session);
            }
        }

        public void registerLatch(UUID session) {
            outputLatches.put(session, new CountDownLatch(1));
        }

        public ByteString waitForOutput(UUID session) {
            CountDownLatch latch = outputLatches.get(session);
            if (latch == null) throw new RuntimeException("No latch registered for session " + session);

            Uninterruptibles.awaitUninterruptibly(latch);
            ByteString output = outputs.get(session);
            if (output == null) throw new RuntimeException("No output received for session " + session);

            outputs.remove(session);
            return output;
        }

        @Override
        public void onError(Throwable throwable) {
            logger.error("Error encountered from worker {}", workerId, throwable);
        }

        @Override
        public void onCompleted() {
            logger.info("Worker {} completed", workerId);
        }

        public CountDownLatch isReady() {
            return readyLatch;
        }
    }
}
