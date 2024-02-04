package com.github.tjake.jlama.net;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.LayerNorm;
import com.github.tjake.jlama.model.TransformerBlock;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.net.grpc.JlamaService;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.TensorInfo;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.protobuf.ByteString;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.testing.GrpcCleanupRule;
import org.junit.Rule;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.Future;

import static org.assertj.core.api.Assertions.assertThat;

public class JlamaServiceTest {
    JlamaServiceGrpc.JlamaServiceBlockingStub blockingStub;
    JlamaServiceGrpc.JlamaServiceStub stub;

    @Rule
    public final GrpcCleanupRule grpcCleanup = new GrpcCleanupRule();

    private final MockConfig modelConfig = new MockConfig(128, 4096, 8192, 16, 12, 1e5f);

    @BeforeEach
    public void setup() throws Exception{

        String serverName = InProcessServerBuilder.generateName();

        grpcCleanup.register(InProcessServerBuilder.forName(serverName)
                .directExecutor()
                .addService(new JlamaService(new MockModel(modelConfig), 4))
                .build()
                .start());

        blockingStub = JlamaServiceGrpc.newBlockingStub(grpcCleanup.register(InProcessChannelBuilder.forName(serverName)
                .directExecutor()
                .build()));

        stub = JlamaServiceGrpc.newStub(grpcCleanup.register(InProcessChannelBuilder.forName(serverName)
                .directExecutor()
                .build()));
    }

    @Test
    public void testRegister() {
        UUID uuid = UUID.randomUUID();
        RegisterRequest request = RegisterRequest.newBuilder().setWorkerid(
                ByteString.copyFrom(ByteBuffer.allocate(128).putLong(uuid.getLeastSignificantBits()).putLong(uuid.getMostSignificantBits()).flip())).build();
        RegisterResponse response = blockingStub.register(request);
        assertThat(response.getOffset()).isEqualTo(0);
        assertThat(response.getLength()).isEqualTo(1024);

        // Should get the same response if we register again with same uuid
        response = blockingStub.register(request);
        assertThat(response.getOffset()).isEqualTo(0);
        assertThat(response.getLength()).isEqualTo(1024);


        // Should get a different response if we register with a different uuid
        uuid = UUID.randomUUID();
        request = RegisterRequest.newBuilder().setWorkerid(
                ByteString.copyFrom(ByteBuffer.allocate(128).putLong(uuid.getLeastSignificantBits()).putLong(uuid.getMostSignificantBits()).flip())).build();
        response = blockingStub.register(request);
        assertThat(response.getOffset()).isEqualTo(1024);
        assertThat(response.getLength()).isEqualTo(1024);
    }


    @Test
    public void testNorm() {
        UUID uuid = UUID.randomUUID();
        NormRequest request = NormRequest.newBuilder()
                .setUuid(ByteString.copyFrom(ByteBuffer.allocate(128).putLong(uuid.getLeastSignificantBits()).putLong(uuid.getMostSignificantBits()).flip()))
                .setLayer(0)
                .setSumSq(10)
                .build();

        /*ListenableFuture<NormResponse> response1 = futureStub.norm(request);
        ListenableFuture<NormResponse> response2 = futureStub.norm(request);
        ListenableFuture<NormResponse> response3 = futureStub.norm(request);
        ListenableFuture<NormResponse> response4 = futureStub.norm(request);

        Futures.whenAllComplete(response1, response2, response3, response4);
        assertThat(response1.resultNow().getSumSq()).isEqualTo(40);
        assertThat(response2.resultNow().getSumSq()).isEqualTo(40);
        assertThat(response3.resultNow().getSumSq()).isEqualTo(40);
        assertThat(response4.resultNow().getSumSq()).isEqualTo(40);*/
    }

    class MockConfig extends Config {
        public MockConfig(int contextLength, int embeddingLength, int hiddenLength, int numberOfHeads, int numberOfLayers, float layerNormEps) {
            super(contextLength, embeddingLength, hiddenLength, numberOfHeads, numberOfHeads, numberOfLayers, layerNormEps, 32000, 1, 2);
        }
    }

    class MockWeightLoader implements WeightLoader {
        @Override
        public Map<String, String> metadata() {
            return Collections.emptyMap();
        }

        @Override
        public Map<String, TensorInfo> tensorInfoMap() {
            return Collections.emptyMap();
        }

        @Override
        public AbstractTensor load(String name, Optional<Pair<Integer, Integer>> offset) {
            return null;
        }


        @Override
        public DType getModelDType() {
            return DType.F32;
        }

        @Override
        public void close() throws Exception {

        }
    }

    class MockTokenizer implements Tokenizer {

        @Override
        public List<String> tokenize(String sentence) {
            return Collections.emptyList();
        }

        @Override
        public long[] encode(String sentence) {
            return new long[0];
        }

        @Override
        public String decode(long id) {
            return "null";
        }

        @Override
        public String decode(long[] ids) {
            return "null";
        }
    }

    class MockModel extends AbstractModel {
        protected MockModel(Config c) {
            super(InferenceType.INPUT_TO_EMBEDDING, c, new MockWeightLoader(), new MockTokenizer(), DType.F32, DType.F32, Optional.empty());
        }

        @Override
        protected EmbedInput loadInputWeights() {
            return null;
        }

        @Override
        protected TransformerBlock[] loadTransformerBlockWeights() {
            return new TransformerBlock[0];
        }

        @Override
        protected SampleOutput loadOutputWeights() {
            return null;
        }
    }


}
