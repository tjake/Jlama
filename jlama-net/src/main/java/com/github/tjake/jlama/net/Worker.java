package com.github.tjake.jlama.net;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.protobuf.ByteString;
import io.grpc.Channel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;

import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

public class Worker {
    private static final Logger logger = org.slf4j.LoggerFactory.getLogger(Worker.class);
    private final UUID workerId;

    private final ByteString workerIdBytes;
    private final AbstractModel model;
    private final JlamaServiceGrpc.JlamaServiceStub client;
    private final JlamaServiceGrpc.JlamaServiceBlockingStub blockingClient;
    private final RegisterResponse registerResponse;

    public Worker(Path modelPrefix, String host, int port, DType workingMemoryType, DType workingQuantizationType, Optional<DType> modelQuantization) {
        Channel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
        this.workerId = UUID.randomUUID();
        this.client = JlamaServiceGrpc.newStub(channel);
        this.blockingClient = JlamaServiceGrpc.newBlockingStub(channel);
        this.workerIdBytes = ByteString.copyFrom(ByteBuffer.allocate(128).putLong(workerId.getMostSignificantBits()).putLong(workerId.getLeastSignificantBits()).flip());
        this.registerResponse = blockingClient.register(RegisterRequest.newBuilder().setWorkerid(workerIdBytes).build());
        logger.info("Registered worker {} with offset {} and length {}", workerId, registerResponse.getOffset(), registerResponse.getLength());
        this.model = loadModel(AbstractModel.InferenceType.FORWARD_PASS, modelPrefix.toFile(), workingMemoryType, workingQuantizationType, modelQuantization, Optional.empty(), Optional.of(Pair.create(registerResponse.getOffset(), registerResponse.getLength())));
    }

    class GenerateObserver implements StreamObserver<GenerateResponse> {
        private final CountDownLatch finishedLatch;
        private final ConcurrentMap<UUID, AbstractTensor> kvBufferCache;
        private final ConcurrentMap<UUID, AtomicInteger> requestCount = new ConcurrentHashMap<>();

        private volatile StreamObserver<GenerateRequest> outputStream;

        private GenerateObserver(CountDownLatch finishedLatch) {
            this.finishedLatch = finishedLatch;
            this.kvBufferCache = new ConcurrentHashMap<>();
        }

        private AbstractTensor getKvBuffer(UUID session) {
            return kvBufferCache.computeIfAbsent(session, s -> model.makeTensor(model.getConfig().getNumberOfLayers(), model.getConfig().contextLength, 2, model.getConfig().embeddingLength));
        }

        private int getNextRequestCount(UUID session) {
            return requestCount.computeIfAbsent(session, s -> new AtomicInteger(0)).incrementAndGet();
        }

        @Override
        public void onNext(GenerateResponse generateResponse) {
            int token = generateResponse.getToken();
            int position = generateResponse.getPosition();
            ByteBuffer bb = generateResponse.getSession().asReadOnlyByteBuffer();
            UUID session = new UUID(bb.getLong(), bb.getLong());

            logger.info("Processing token {} at position {} for session {}", token, position, session);

            AbstractTensor output = model.forward(token, position, getKvBuffer(session), Optional.of((a, b) -> {
                NormRequest nr = NormRequest.newBuilder().setUuid(generateResponse.getSession()).setWorkerid(workerIdBytes).setLayer(getNextRequestCount(session)).setSumSq(a).setSum(b).build();
                NormResponse normResponse = blockingClient.norm(nr);
                return Pair.create(normResponse.getSumSq(), normResponse.getSum());
            }));

            ByteString tensor = ByteString.copyFrom(output.getMemorySegment().toArray(ValueLayout.JAVA_BYTE));
            output.close();

            outputStream.onNext(GenerateRequest.newBuilder()
                    .setSession(generateResponse.getSession())
                    .setWorkerid(workerIdBytes)
                    .setTensor(tensor)
                    .build());
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

        //Request first token
        request.onNext(GenerateRequest.newBuilder().setWorkerid(workerIdBytes).build());

        Uninterruptibles.awaitUninterruptibly(finishedLatch);
    }
}
