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

import static org.assertj.core.api.Assertions.assertThat;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.DistributedContext;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.TransformerBlock;
import com.github.tjake.jlama.model.functions.EmbedInput;
import com.github.tjake.jlama.model.functions.SampleOutput;
import com.github.tjake.jlama.net.grpc.JlamaService;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.TensorInfo;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.safetensors.tokenizer.TokenizerModel;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.google.protobuf.ByteString;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.testing.GrpcCleanupRule;
import java.nio.ByteBuffer;
import java.util.*;
import org.junit.Rule;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class JlamaServiceTest {
    JlamaServiceGrpc.JlamaServiceBlockingStub blockingStub;
    JlamaServiceGrpc.JlamaServiceStub stub;

    @Rule
    public final GrpcCleanupRule grpcCleanup = new GrpcCleanupRule();

    private final MockConfig modelConfig = new MockConfig(128, 4096, 8192, 16, 12, 1e5f);

    @BeforeEach
    public void setup() throws Exception {

        String serverName = InProcessServerBuilder.generateName();

        grpcCleanup.register(
            InProcessServerBuilder.forName(serverName)
                .directExecutor()
                .addService(new JlamaService(new MockModel(modelConfig), 4, true, false))
                .build()
                .start()
        );

        blockingStub = JlamaServiceGrpc.newBlockingStub(
            grpcCleanup.register(InProcessChannelBuilder.forName(serverName).directExecutor().build())
        );

        stub = JlamaServiceGrpc.newStub(grpcCleanup.register(InProcessChannelBuilder.forName(serverName).directExecutor().build()));
    }

    @Test
    public void testRegister() {
        UUID uuid = UUID.randomUUID();
        RegisterRequest request = RegisterRequest.newBuilder()
            .setWorkerid(
                ByteString.copyFrom(
                    ByteBuffer.allocate(128).putLong(uuid.getLeastSignificantBits()).putLong(uuid.getMostSignificantBits()).flip()
                )
            )
            .build();
        RegisterResponse response = blockingStub.register(request);
        assertThat(response.getModelShard()).isEqualTo(0);
        assertThat(response.getNumModelShards()).isEqualTo(4);

        // Should get the same response if we register again with same uuid
        response = blockingStub.register(request);
        assertThat(response.getModelShard()).isEqualTo(0);
        assertThat(response.getNumModelShards()).isEqualTo(4);

        // Should get a different response if we register with a different uuid
        uuid = UUID.randomUUID();
        request = RegisterRequest.newBuilder()
            .setWorkerid(
                ByteString.copyFrom(
                    ByteBuffer.allocate(128).putLong(uuid.getLeastSignificantBits()).putLong(uuid.getMostSignificantBits()).flip()
                )
            )
            .build();
        response = blockingStub.register(request);
        assertThat(response.getModelShard()).isEqualTo(1);
        assertThat(response.getNumModelShards()).isEqualTo(4);
    }

    public static class MockConfig extends Config {
        public MockConfig(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfLayers,
            float layerNormEps
        ) {
            super(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfHeads,
                numberOfLayers,
                layerNormEps,
                32000,
                1,
                List.of(2),
                ActivationFunction.Type.SILU,
                10000.0,
                1.0
            );
        }
    }

    public static class MockWeightLoader implements WeightLoader {
        @Override
        public Map<String, String> metadata() {
            return Collections.emptyMap();
        }

        @Override
        public Map<String, TensorInfo> tensorInfoMap() {
            return Collections.emptyMap();
        }

        @Override
        public AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns) {
            return null;
        }

        @Override
        public DType getModelDType() {
            return DType.F32;
        }

        @Override
        public void close() throws Exception {}
    }

    public static class MockTokenizer implements Tokenizer {

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

        @Override
        public Optional<PromptSupport> promptSupport() {
            return Optional.empty();
        }

        @Override
        public TokenizerModel getModel() {
            return null;
        }
    }

    public static class MockModel extends AbstractModel {
        public MockModel(Config c) {
            super(InferenceType.INPUT_TO_EMBEDDING, c, new MockWeightLoader(), new MockTokenizer(), DType.F32, DType.F32, Optional.empty());
        }

        @Override
        public ModelSupport.ModelType getModelType() {
            return ModelSupport.getModelType("LLAMA");
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
