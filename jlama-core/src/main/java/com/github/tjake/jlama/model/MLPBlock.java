package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

import java.util.Optional;

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
    public AbstractTensor forward(AbstractTensor lnemb) {
        int hiddenLength = model.c.hiddenLength;
        try(AbstractTensor buf = model.makeTensor(hiddenLength); AbstractTensor buf2 = model.makeTensor(hiddenLength)) {

            VectorMath.pchunk(model.c.hiddenLength, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get().dotProductChunk(buf, lnemb, fullyConnectedWeights, model.c.embeddingLength, chunkStart, chunkSize);

                if (upProjectionWeights != null) {
                    TensorOperationsProvider.get().dotProductChunk(buf2, lnemb, upProjectionWeights, model.c.embeddingLength, chunkStart, chunkSize);
                }
            });

            fullyConnectedBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(buf, bias));

            VectorMath.pfor(0, hiddenLength, i -> {
                float w1 = buf.get(i);
                float w1a = ActivationFunction.eval(activationFunction, w1);
                buf.set(w1a, i);
            });

            if (upProjectionWeights != null) {
                TensorOperationsProvider.get().maccumulate(buf, buf2);
            }

            //matmul the projection and sum into input
            AbstractTensor result = model.makeTensor(model.c.embeddingLength);
            VectorMath.pchunk(model.c.embeddingLength, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get().dotProductChunk(result, buf, projectionWeights, hiddenLength, chunkStart, chunkSize);
            });
            projectionBias.ifPresent(bias -> TensorOperationsProvider.get().accumulate(result, bias));
            return result;
        }
    }
}
