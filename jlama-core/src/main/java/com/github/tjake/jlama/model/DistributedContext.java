package com.github.tjake.jlama.model;

import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class DistributedContext {

    private final Config c;
    private final int modelShard;
    private final int numModelShards;
    private final int layerShard;
    private final int numLayerShards;

    private final Consumer<List<AbstractTensor>> tensorSync;

    // Suppliers to store values that chance when offset is adjusted
    public final int embeddingSegmentStart;
    public final int embeddingSegmentLength;
    public final int embeddingSegmentEnd;

    public final int attentionSegmentStart;
    public final int attentionSegmentLength;
    public final int attentionSegmentEnd;

    public final int hiddenSegmentStart;
    public final int hiddenSegmentLength;
    public final int hiddenSegmentEnd;

    public final int kvSegmentStart;
    public final int kvSegmentLength;
    public final int kvSegmentEnd;

    public final int headStart;
    public final int headEnd;
    public final int groupHeadStart;
    public final int groupHeadEnd;

    public final int numberOfLayers;
    public final int layerStart;
    public final int layerEnd;

    private DistributedContext(Config c, int modelShard, int numModelShards, int layerShard, int numLayerShards, Consumer<List<AbstractTensor>> tensorSync) {
        this.c = c;
        this.modelShard = modelShard;
        this.numModelShards = numModelShards;
        this.layerShard = layerShard;
        this.numLayerShards = numLayerShards;
        this.tensorSync = tensorSync;

        this.numberOfLayers = c.numberOfLayers / numLayerShards;
        this.layerStart = numberOfLayers * layerShard;
        this.layerEnd = layerStart + numberOfLayers;

        this.embeddingSegmentLength = c.embeddingLength / numModelShards;
        this.embeddingSegmentStart = embeddingSegmentLength * modelShard;
        this.embeddingSegmentEnd = embeddingSegmentStart + embeddingSegmentLength;

        this.attentionSegmentLength = c.attentionLength / numModelShards;
        this.attentionSegmentStart = attentionSegmentLength * modelShard;
        this.attentionSegmentEnd = attentionSegmentStart + attentionSegmentLength;

        this.hiddenSegmentLength = c.hiddenLength / numModelShards;
        this.hiddenSegmentStart = hiddenSegmentLength * modelShard;
        this.hiddenSegmentEnd = hiddenSegmentStart + hiddenSegmentLength;

        this.kvSegmentStart = embeddingSegmentStart / c.headGroupSize;
        this.kvSegmentEnd = embeddingSegmentEnd / c.headGroupSize;
        this.kvSegmentLength = embeddingSegmentLength / c.headGroupSize;


        this.headStart = embeddingSegmentStart / c.headSize;
        this.headEnd = embeddingSegmentEnd / c.headSize;
        this.groupHeadStart = kvSegmentStart / c.headSize;
        this.groupHeadEnd = kvSegmentEnd / c.headSize;
    }

    public boolean hasModelShard() {
        return numModelShards > 1;
    }

    public void syncTensors(List<AbstractTensor> tensors) {
        tensorSync.accept(tensors);
    }

    public int getShardOffsetForLength(int length) {
        return length / numModelShards * modelShard;
    }

    public int getShardLength(int length) {
        return length / numModelShards;
    }

    public static Builder builder(Config c) {
        return new Builder(c);
    }

    public static class Builder {
        private Config c;
        private int modelShard = 0;
        private int numModelShards = 1;
        private int layerShard = 0;
        private int numLayerShards = 1;
        private Consumer<List<AbstractTensor>> tensorSync;

        public Builder(Config c) {
            this.c = c;
        }

        public Builder setModelShard(int modelShard) {
            this.modelShard = modelShard;
            return this;
        }

        public Builder setNumModelShards(int numModelShards) {
            this.numModelShards = numModelShards;
            return this;
        }

        public Builder setLayerShard(int layerShard) {
            this.layerShard = layerShard;
            return this;
        }

        public Builder setNumLayerShards(int numLayerShards) {
            this.numLayerShards = numLayerShards;
            return this;
        }

        public Builder setTensorSync(Consumer<List<AbstractTensor>> tensorSync) {
            this.tensorSync = tensorSync;
            return this;
        }

        public DistributedContext build() {
            return new DistributedContext(c, modelShard, numModelShards, layerShard, numLayerShards, tensorSync);
        }
    }
}
