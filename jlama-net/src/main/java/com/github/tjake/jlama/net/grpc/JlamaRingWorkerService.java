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

import com.github.tjake.jlama.net.*;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.TensorShape;
import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class JlamaRingWorkerService extends JlamaWorkerRingGrpc.JlamaWorkerRingImplBase {

    private static final Logger logger = LoggerFactory.getLogger(JlamaRingWorkerService.class);

    private final Worker worker;

    public JlamaRingWorkerService(Worker worker) {
        this.worker = worker;
    }

    @Override
    public StreamObserver<PassRecord> pass(StreamObserver<Empty> responseObserver) {

        return new StreamObserver<>() {
            @Override
            public void onNext(PassRecord value) {
                // logger.info("Received pass record from peer");
                int startPosition = value.getStartPosition();
                FloatBuffer buffer = value.getTensor().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                AbstractTensor tensor = new FloatBufferTensor(
                    buffer,
                    TensorShape.of(value.getBatchSize(), worker.model.getConfig().embeddingLength),
                    true
                );
                ByteString sessionBytes = value.getSession();

                worker.pass(sessionBytes, startPosition, tensor);
            }

            @Override
            public void onError(Throwable t) {
                logger.error("Received error from peer", t);
            }

            @Override
            public void onCompleted() {
                responseObserver.onNext(Empty.newBuilder().build());
                responseObserver.onCompleted();
            }
        };

    }
}
