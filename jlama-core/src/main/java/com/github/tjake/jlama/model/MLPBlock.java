package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;

public class MLPBlock {
    private final AbstractModel model;
    private final Optional<AbstractTensor> fullyConnectedBias;
    private final AbstractTensor fullyConnectedWeights;

    private final Optional<AbstractTensor> projectionBias;
    private final AbstractTensor projectionWeights;

    private final AbstractTensor upProjectionWeights;

    private final ActivationFunction.Type activationFunction;


    public MLPBlock(AbstractModel model, ActivationFunction.Type activationFunction, AbstractTensor fullyConnectedBias, AbstractTensor fullyConnectedWeights, AbstractTensor projectionBias, AbstractTensor projectionWeights)
    {
        this(model, activationFunction, Optional.of(fullyConnectedBias), fullyConnectedWeights, Optional.of(projectionBias), projectionWeights, null);
    }

    public MLPBlock(AbstractModel model, ActivationFunction.Type activationFunction, AbstractTensor fullyConnectedWeights, AbstractTensor projectionWeights, AbstractTensor upProjectionWeights) {
        this(model, activationFunction, Optional.empty(), fullyConnectedWeights, Optional.empty(), projectionWeights, upProjectionWeights);
    }

    public MLPBlock(AbstractModel model, ActivationFunction.Type activationFunction, Optional<AbstractTensor> fullyConnectedBias, AbstractTensor fullyConnectedWeights, Optional<AbstractTensor> projectionBias, AbstractTensor projectionWeights, AbstractTensor upProjectionWeights)
    {
        this.model = model;
        this.activationFunction = activationFunction;
        this.fullyConnectedBias = fullyConnectedBias;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionBias = projectionBias;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = upProjectionWeights;
    }

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    public AbstractTensor forward(AbstractTensor lnemb, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int hiddenLength = model.c.hiddenLength;
        try(AbstractTensor buf = model.makeTensor(hiddenLength); AbstractTensor buf2 = model.makeTensor(hiddenLength)) {

            VectorMath.pchunk(0, hiddenLength, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get().dotProductChunk(buf, lnemb, fullyConnectedWeights, model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength(), chunkStart, chunkSize);

                if (upProjectionWeights != null) {
                    TensorOperationsProvider.get().dotProductChunk(buf2, lnemb, upProjectionWeights, model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength(), chunkStart, chunkSize);
                }
            });

            tensorReducer.ifPresent(func -> {
                List<AbstractTensor> ts = new ArrayList<>(2);
                ts.add(buf);
                if (upProjectionWeights != null) ts.add(buf2);

                func.accept(ts);
            });

            fullyConnectedBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(buf, bias, 0, hiddenLength));

            VectorMath.pfor(0, hiddenLength, i -> {
                float w1 = buf.get(i);
                float w1a = ActivationFunction.eval(activationFunction, w1);
                buf.set(w1a, i);
            });

            if (upProjectionWeights != null) {
                TensorOperationsProvider.get().maccumulate(buf, buf2, 0, hiddenLength);
            }

            //matmul the projection and sum into input
            AbstractTensor result = model.makeTensor(model.c.embeddingLength);
            VectorMath.pchunk(model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength(), (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get().dotProductChunk(result, buf, projectionWeights, 0, hiddenLength, chunkStart, chunkSize);
            });
            projectionBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(result, bias, model.c.embeddingSegmentStart(), model.c.embeddingSegmentLength()));
            return result;
        }
    }
}
