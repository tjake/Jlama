package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;

public class MLPBlock {
    private final AbstractModel model;
    private final AbstractTensor fullyConnectedBias;
    private final AbstractTensor fullyConnectedWeights;

    private final AbstractTensor projectionBias;
    private final AbstractTensor projectionWeights;

    private final AbstractTensor upProjectionWeights;

    private final ActivationFunction.Type activationFunction;

    public MLPBlock(AbstractModel model, ActivationFunction.Type activationFunction, AbstractTensor fullyConnectedBias, AbstractTensor fullyConnectedWeights, AbstractTensor projectionBias, AbstractTensor projectionWeights)
    {
        this(model, activationFunction, fullyConnectedBias, fullyConnectedWeights, projectionBias, projectionWeights, null);
    }

    public MLPBlock(AbstractModel model, ActivationFunction.Type activationFunction, AbstractTensor fullyConnectedBias, AbstractTensor fullyConnectedWeights, AbstractTensor projectionBias, AbstractTensor projectionWeights, AbstractTensor upProjectionWeights)
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
        try(AbstractTensor buf = model.makeTensor(hiddenLength)) {
            VectorMath.pfor(0, hiddenLength, i -> {
                float w1 = fullyConnectedBias.get(i) + TensorOperationsProvider.get().dotProduct(lnemb, fullyConnectedWeights.slice(i), model.c.embeddingLength);
                float w1a = ActivationFunction.eval(activationFunction, w1);

                if (upProjectionWeights != null) {
                    float w3 = TensorOperationsProvider.get().dotProduct(lnemb, upProjectionWeights.slice(i), model.c.embeddingLength);
                    w1a *= w3;
                }

                buf.set(w1a, i);
            });

            //matmul the projection and sum into input
            AbstractTensor result = model.makeTensor(model.c.embeddingLength);
            VectorMath.pfor(0, model.c.embeddingLength, i -> {
                float v = projectionBias.get(i) + TensorOperationsProvider.get().dotProduct(buf, projectionWeights.slice(i), hiddenLength);
                result.set(v, i);
            });

            return result;
        }
    }
}
