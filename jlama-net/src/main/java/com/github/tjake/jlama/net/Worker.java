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
import com.github.tjake.jlama.net.grpc.JlamaRingWorkerService;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.HTTPSafeTensorLoader;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.KvBufferCache;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.protobuf.ByteString;
import com.google.protobuf.UnsafeByteOperations;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.io.*;
import java.lang.foreign.MemorySegment;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Worker implements Closeable {

    private static final Integer MESSAGE_SIZE = 1024 * 1024 * 1024;
    private static final String HOSTNAME;
    static {
        try {
            HOSTNAME = System.getenv("HOSTNAME") == null ? InetAddress.getLocalHost().getHostName() : System.getenv("HOSTNAME");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static final Logger logger = LoggerFactory.getLogger(Worker.class);
    private final UUID workerId;
    private final KvBufferCache kvBufferCache;
    private final ByteString workerIdBytes;
    public final AbstractModel model;
    private final JlamaServiceGrpc.JlamaServiceStub client;
    private final JlamaServiceGrpc.JlamaServiceBlockingStub blockingClient;
    private final RegisterResponse registerResponse;

    public final PeerInfo peerInfo;
    private final JlamaWorkerRingGrpc.JlamaWorkerRingStub peerClient;
    private final StreamObserver<PassRecord> peerStream;

    private volatile StreamObserver<GenerateRequest> outputStream;
    private final ConcurrentMap<UUID, CombineObserver> combineStreams;

    private final Server peerServer;
    private final JlamaRingWorkerService peerService;

    public Worker(
        File modelPath,
        String modelOwner,
        String modelName,
        DType modelDType,
        String host,
        int coordinatorPort,
        int peerPort,
        File workingDirectory,
        DType workingMemoryType,
        DType workingQuantizationType,
        Optional<DType> modelQuantization,
        Optional<String> optionalWorkerId,
        Optional<String> authToken,
        Optional<String> branch
    ) {
        Channel channel = ManagedChannelBuilder.forAddress(host, coordinatorPort)
            .usePlaintext()
            .maxInboundMessageSize(MESSAGE_SIZE)
            .build();

        // Start the ring service
        this.peerService = new JlamaRingWorkerService(this);
        this.peerServer = ServerBuilder.forPort(peerPort).addService(peerService).maxInboundMessageSize(MESSAGE_SIZE).build();
        try {
            this.peerServer.start();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Setup via coordinator
        this.workerId = optionalWorkerId.map(s -> new UUID(s.hashCode(), s.hashCode())).orElse(UUID.randomUUID());
        this.client = JlamaServiceGrpc.newStub(channel).withMaxInboundMessageSize(MESSAGE_SIZE).withMaxOutboundMessageSize(MESSAGE_SIZE);
        this.blockingClient = JlamaServiceGrpc.newBlockingStub(channel)
            .withMaxInboundMessageSize(MESSAGE_SIZE)
            .withMaxOutboundMessageSize(MESSAGE_SIZE);
        this.workerIdBytes = ByteString.copyFrom(
            ByteBuffer.allocate(128).putLong(workerId.getMostSignificantBits()).putLong(workerId.getLeastSignificantBits()).flip()
        );

        RegisterRequest rr = RegisterRequest.newBuilder().setWorkerid(workerIdBytes).setHostname(HOSTNAME).setPeerPort(peerPort).build();

        this.registerResponse = blockingClient.register(rr);

        logger.info(
            "Registered worker {} with model shard {} of {}, layer shard {} of {}",
            workerId,
            registerResponse.getModelShard(),
            registerResponse.getNumModelShards(),
            registerResponse.getLayerShard(),
            registerResponse.getNumLayerShards()
        );

        // Setup peer
        this.peerInfo = registerResponse.getNumLayerShards() == 1 ? null : blockingClient.discover(rr);
        this.peerClient = peerInfo == null || peerInfo.getIsCoordinator()
            ? null
            : JlamaWorkerRingGrpc.newStub(
                ManagedChannelBuilder.forAddress(peerInfo.getHostname(), peerInfo.getPeerPort()).usePlaintext().build()
            );
        this.peerStream = peerInfo == null || peerInfo.getIsCoordinator() ? null : peerClient.pass(new StreamObserver<>() {
            @Override
            public void onNext(Empty empty) {}

            @Override
            public void onError(Throwable throwable) {
                logger.error("Error in peer", throwable);
            }

            @Override
            public void onCompleted() {
                logger.info("PeerResponseStream completed");
            }
        });

        this.combineStreams = new ConcurrentHashMap<>();

        // Load the model
        Function<File, WeightLoader> weightLoaderFunction = SafeTensorSupport.isModelLocal(modelPath.toPath())
            ? b -> SafeTensorSupport.loadWeights(modelPath)
            : b -> new HTTPSafeTensorLoader(modelPath.toPath(), modelOwner, modelName, modelDType, authToken, branch);

        this.model = loadModel(
            AbstractModel.InferenceType.FORWARD_PASS,
            modelPath,
            workingDirectory,
            workingMemoryType,
            workingQuantizationType,
            modelQuantization,
            Optional.empty(),
            Optional.of(
                c -> DistributedContext.builder(c)
                    .setModelShard(registerResponse.getModelShard())
                    .setNumModelShards(registerResponse.getNumModelShards())
                    .setLayerShard(registerResponse.getLayerShard())
                    .setNumLayerShards(registerResponse.getNumLayerShards())
                    .build()
            ),
            weightLoaderFunction
        );

        this.kvBufferCache = new KvBufferCache(model);

        logger.info("Model loaded");
        logger.info(model.getConfig().dctx().toString());
    }

    private CombineObserver getCombineResponseStream(UUID session) {
        return combineStreams.computeIfAbsent(session, s -> new CombineObserver(session));
    }

    private ByteString getTensorBytes(AbstractTensor tensor) {
        Preconditions.checkArgument(tensor.dims() == 2 && tensor.dType() == DType.F32);
        return UnsafeByteOperations.unsafeWrap(tensor.getMemorySegment().asByteBuffer());
    }

    public void pass(ByteString sessionBytes, int startPosition, AbstractTensor tensor) {
        ByteBuffer bb = sessionBytes.asReadOnlyByteBuffer();
        UUID session = new UUID(bb.getLong(), bb.getLong());

        // logger.info("From Peer: {} token(s) from position {} for session {}", tensor.shape().first(), startPosition, session);

        Consumer<List<AbstractTensor>> combineCallback = registerResponse.getNumModelShards() == 1 ? t -> {} : t -> {
            CombineRequest.Builder nrb = CombineRequest.newBuilder()
                .setUuid(sessionBytes)
                .setWorkerid(workerIdBytes)
                .setLayerShard(registerResponse.getLayerShard())
                .setModelShard(registerResponse.getModelShard());

            for (int i = 0; i < t.size(); i++)
                nrb = nrb.addTensor(getTensorBytes(t.get(i)));

            // logger.info("1)Sending combine request for session {}", session);
            CombineResponse combineResponse = getCombineResponseStream(session).request(nrb.build()).join();

            for (int i = 0; i < t.size(); i++)
                t.get(i)
                    .getMemorySegment()
                    .copyFrom(MemorySegment.ofBuffer(combineResponse.getTensor(i).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)));
        };

        AbstractTensor output = model.forward(tensor, startPosition, kvBufferCache.getKvBuffer(session), Optional.of(combineCallback));

        processOutput(sessionBytes, startPosition, tensor.shape().first(), output);
    }

    public void processOutput(ByteString session, int startPosition, int batchSize, AbstractTensor output) {
        if (peerInfo == null || peerInfo.getIsCoordinator()) {
            outputStream.onNext(
                GenerateRequest.newBuilder()
                    .setSession(session)
                    .setWorkerid(workerIdBytes)
                    .setTensor(getTensorBytes(output.slice(output.shape().first() - 1))) // keep only the last token
                    .build()
            );
        } else {
            // Send the last token to the next worker
            PassRecord peerRequest = PassRecord.newBuilder()
                .setSession(session)
                .setStartPosition(startPosition)
                .setBatchSize(batchSize)
                .setTensor(getTensorBytes(output))
                .build();

            peerStream.onNext(peerRequest);
        }

        output.close();
    }

    @Override
    public void close() {
        try {
            kvBufferCache.close();
        } catch (Exception e) {
            logger.error("Error closing kvBufferCache", e);
        }

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

        private volatile StreamObserver<GenerateRequest> outputStream;

        private GenerateObserver(CountDownLatch finishedLatch) {
            this.finishedLatch = finishedLatch;
        }

        @Override
        public void onNext(GenerateResponse generateResponse) {
            int[] tokens = generateResponse.getTokensList().stream().mapToInt(Integer::intValue).toArray();
            int startPosition = generateResponse.getStartPosition();
            ByteBuffer bb = generateResponse.getSession().asReadOnlyByteBuffer();
            UUID session = new UUID(bb.getLong(), bb.getLong());

            // logger.info("From Coordinator: {} token(s) from position {} for session {}", tokens.length,
            // startPosition, session);

            Consumer<List<AbstractTensor>> combineCallback = registerResponse.getNumModelShards() == 1 ? t -> {} : t -> {
                CombineRequest.Builder nrb = CombineRequest.newBuilder()
                    .setUuid(generateResponse.getSession())
                    .setWorkerid(workerIdBytes)
                    .setLayerShard(registerResponse.getLayerShard())
                    .setModelShard(registerResponse.getModelShard());
                for (int i = 0; i < t.size(); i++)
                    nrb = nrb.addTensor(getTensorBytes(t.get(i)));

                // logger.info("2){} Sending combine request for session {}", registerResponse.getWorkerOrd(), session);

                CombineResponse combineResponse = getCombineResponseStream(session).request(nrb.build()).join();

                for (int i = 0; i < t.size(); i++)
                    t.get(i)
                        .getMemorySegment()
                        .copyFrom(
                            MemorySegment.ofBuffer(combineResponse.getTensor(i).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN))
                        );
            };

            AbstractTensor output = model.batchForward(
                tokens,
                startPosition,
                kvBufferCache.getKvBuffer(session),
                Optional.of(combineCallback)
            );

            processOutput(generateResponse.getSession(), startPosition, tokens.length, output);
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
        this.outputStream = client.generate(observer);
        observer.setOutputStream(outputStream);

        // Request first token
        outputStream.onNext(GenerateRequest.newBuilder().setWorkerid(workerIdBytes).build());

        Uninterruptibles.awaitUninterruptibly(finishedLatch);

        // Cleanup
        if (peerStream != null) peerStream.onCompleted();
        if (peerClient != null) ((ManagedChannel) peerClient.getChannel()).shutdown();
        peerServer.shutdown();
    }
}
