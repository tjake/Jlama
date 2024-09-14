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
package com.github.tjake.jlama.net;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.DistributedContext;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.protobuf.ByteString;
import com.google.protobuf.UnsafeByteOperations;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;
import java.io.*;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;

public class Worker implements Closeable {
    private static final Logger logger = org.slf4j.LoggerFactory.getLogger(Worker.class);
    private final UUID workerId;
    private final ByteString workerIdBytes;
    private final AbstractModel model;
    private final JlamaServiceGrpc.JlamaServiceStub client;
    private final JlamaServiceGrpc.JlamaServiceBlockingStub blockingClient;
    private final RegisterResponse registerResponse;

    public Worker(
        File modelPrefix,
        String host,
        int port,
        File workingDirectory,
        DType workingMemoryType,
        DType workingQuantizationType,
        Optional<DType> modelQuantization,
        Optional<String> optionalWorkerId
    ) {
        Channel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();

        this.workerId = optionalWorkerId.map(s -> new UUID(s.hashCode(), s.hashCode())).orElse(UUID.randomUUID());
        this.client = JlamaServiceGrpc.newStub(channel);
        this.blockingClient = JlamaServiceGrpc.newBlockingStub(channel);
        this.workerIdBytes = ByteString.copyFrom(
            ByteBuffer.allocate(128).putLong(workerId.getMostSignificantBits()).putLong(workerId.getLeastSignificantBits()).flip()
        );
        this.registerResponse = blockingClient.register(RegisterRequest.newBuilder().setWorkerid(workerIdBytes).build());
        logger.info(
            "Registered worker {} with shard {} of {}",
            workerId,
            registerResponse.getModelShard(),
            registerResponse.getNumModelShards()
        );


        this.model = loadModel(
            AbstractModel.InferenceType.FORWARD_PASS,
            modelPrefix,
            workingDirectory,
            workingMemoryType,
            workingQuantizationType,
            modelQuantization,
            Optional.empty(),
            Optional.of(c -> DistributedContext.builder(c)
                        .setModelShard(registerResponse.getModelShard())
                        .setNumModelShards(registerResponse.getNumModelShards())
                        .setLayerShard(registerResponse.getLayerShard())
                        .setNumLayerShards(registerResponse.getNumLayerShards())
                        .build()
            )
        );
    }

    @Override
    public void close() {
        ((ManagedChannel) client.getChannel()).shutdown();
    }

    class CombineObserver implements StreamObserver<CombineResponse> {

        private final UUID session;
        private final StreamObserver<CombineRequest> requestStreamObserver;

        private final AtomicReference<CompletableFuture<CombineResponse>> activeRequestFuture;

        CombineObserver(UUID session) {
            this.session = session;
            this.requestStreamObserver = client.combine(this);
            this.activeRequestFuture = new AtomicReference<>();
        }

        public CompletableFuture<CombineResponse> request(CombineRequest request) {
            CompletableFuture<CombineResponse> f = new CompletableFuture<>();
            if (!activeRequestFuture.compareAndSet(null, f)) throw new IllegalStateException(
                "active future still outstanding for " + session
            );

            requestStreamObserver.onNext(request);

            return f;
        }

        @Override
        public void onNext(CombineResponse combineResponse) {
            CompletableFuture<CombineResponse> f = activeRequestFuture.getAndSet(null);
            if (f == null) logger.error("Missing future for {}", session);
            else f.complete(combineResponse);
        }

        @Override
        public void onError(Throwable throwable) {
            CompletableFuture<CombineResponse> f = activeRequestFuture.getAndSet(null);
            if (f == null) logger.error("Missing future for {}", session);
            else f.completeExceptionally(throwable);
        }

        @Override
        public void onCompleted() {
            logger.info("CombineResponseStream {} completed", session);
            CompletableFuture<CombineResponse> f = activeRequestFuture.getAndSet(null);

            if (f != null) f.completeExceptionally(new RuntimeException("Stream was completed for " + session));
        }
    }

    class GenerateObserver implements StreamObserver<GenerateResponse> {
        private final CountDownLatch finishedLatch;
        private final KvBufferCache kvBufferCache;
        private final ConcurrentMap<UUID, AtomicInteger> requestCount;
        private final ConcurrentMap<UUID, CombineObserver> combineStreams;

        private volatile StreamObserver<GenerateRequest> outputStream;

        private GenerateObserver(CountDownLatch finishedLatch) {
            this.finishedLatch = finishedLatch;
            this.kvBufferCache = new KvBufferCache(model);
            this.requestCount = new ConcurrentHashMap<>();
            this.combineStreams = new ConcurrentHashMap<>();
        }

        private int getNextRequestCount(UUID session) {
            return requestCount.computeIfAbsent(session, s -> new AtomicInteger(0)).incrementAndGet();
        }

        private CombineObserver getCombineResponseStream(UUID session) {
            return combineStreams.computeIfAbsent(session, s -> new CombineObserver(session));
        }

        private ByteString getTensorBytes(AbstractTensor tensor) {
            Preconditions.checkArgument(tensor.dims() == 2 && tensor.dType() == DType.F32);
            return UnsafeByteOperations.unsafeWrap(tensor.getMemorySegment().asByteBuffer());
        }

        @Override
        public void onNext(GenerateResponse generateResponse) {
            int token = generateResponse.getToken();
            int position = generateResponse.getPosition();
            ByteBuffer bb = generateResponse.getSession().asReadOnlyByteBuffer();
            UUID session = new UUID(bb.getLong(), bb.getLong());

            logger.info("Processing token {} at position {} for session {}", token, position, session);

            AbstractTensor output = model.forward(token, position, kvBufferCache.getKvBuffer(session), Optional.of((a, b) -> {
               return null;
            }), Optional.of(t -> {
                CombineRequest.Builder nrb = CombineRequest.newBuilder()
                    .setUuid(generateResponse.getSession())
                    .setWorkerid(workerIdBytes)
                    .setLayer(getNextRequestCount(session));
                for (int i = 0; i < t.size(); i++)
                    nrb = nrb.addTensor(getTensorBytes(t.get(i)));

                CombineResponse combineResponse = getCombineResponseStream(session).request(nrb.build()).join();

                for (int i = 0; i < t.size(); i++)
                    t.get(i)
                        .getMemorySegment()
                        .copyFrom(
                            MemorySegment.ofBuffer(combineResponse.getTensor(i).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN))
                        );
            }));

            outputStream.onNext(
                GenerateRequest.newBuilder()
                    .setSession(generateResponse.getSession())
                    .setWorkerid(workerIdBytes)
                    .setTensor(getTensorBytes(output))
                    .build()
            );

            output.close();
        }

        @Override
        public void onError(Throwable throwable) {
            logger.error("Error in generate", throwable);
        }

        @Override
        public void onCompleted() {
            finishedLatch.countDown();
        }

        public void setOutputStream(StreamObserver<GenerateRequest> outputStream) {
            this.outputStream = outputStream;
        }
    }

    public void run() {
        CountDownLatch finishedLatch = new CountDownLatch(1);
        GenerateObserver observer = new GenerateObserver(finishedLatch);
        StreamObserver<GenerateRequest> request = client.generate(observer);
        observer.setOutputStream(request);

        // Request first token
        request.onNext(GenerateRequest.newBuilder().setWorkerid(workerIdBytes).build());

        Uninterruptibles.awaitUninterruptibly(finishedLatch);
    }
}
