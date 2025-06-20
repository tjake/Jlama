package com.github.tjake.jlama.model.gemma;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

public class GemmaModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return GemmaModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return GemmaConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return GemmaTokenizer.class;
    }
}
