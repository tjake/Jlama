package com.github.tjake.jlama.model.mixtral;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

public class MixtralModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return MixtralModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return MixtralConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return LlamaTokenizer.class;
    }
}
