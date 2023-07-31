package com.github.tjake.llmj.model;

import com.github.tjake.llmj.math.ActivationFunction;
import com.github.tjake.llmj.math.VectorMath;
import com.github.tjake.llmj.safetensors.Config;

import java.util.stream.IntStream;

public class MLPBlock {

    private final Config c;
    private final Tensor fullyConnectedBias;
    private final Tensor fullyConnectedWeights;

    private final Tensor projectionBias;
    private final Tensor projectionWeights;

    private final Tensor upProjectionWeights;

    private final ActivationFunction.Type activationFunction;

    public MLPBlock(Config c, ActivationFunction.Type activationFunction, Tensor fullyConnectedBias, Tensor fullyConnectedWeights, Tensor projectionBias, Tensor projectionWeights)
    {
        this(c, activationFunction, fullyConnectedBias, fullyConnectedWeights, projectionBias, projectionWeights, null);
    }

    public MLPBlock(Config c, ActivationFunction.Type activationFunction, Tensor fullyConnectedBias, Tensor fullyConnectedWeights, Tensor projectionBias, Tensor projectionWeights, Tensor fc2)
    {
        this.c = c;
        this.activationFunction = activationFunction;
        this.fullyConnectedBias = fullyConnectedBias;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionBias = projectionBias;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = fc2;
    }

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    public Tensor forward(Tensor lnemb) {
        int hiddenLength = c.hiddenLength;
        try(Tensor buf = c.bufferCache.get(hiddenLength)) {
            IntStream.range(0, hiddenLength).parallel().forEach( i -> {
                float v = fullyConnectedBias.get(i) + VectorMath.dotProduct(lnemb, fullyConnectedWeights.slice(i), c.embeddingLength);
                float a = ActivationFunction.eval(activationFunction, v);

                if (upProjectionWeights != null)
                    a *= VectorMath.dotProduct(lnemb, upProjectionWeights.slice(i), c.embeddingLength);

                buf.set(a, i);
            });

            //matmul the projection and sum into input
            Tensor result = c.bufferCache.get(c.embeddingLength);
            for (int i = 0; i < c.embeddingLength; i++) {
                float v = projectionBias.get(i) + VectorMath.dotProduct(buf, projectionWeights.slice(i), hiddenLength);
                result.set(v, i);
            }

            return result;
        }
    }
}
