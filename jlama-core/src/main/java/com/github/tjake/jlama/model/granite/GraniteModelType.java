package com.github.tjake.jlama.model.granite;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.gpt2.GPT2Tokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

public class GraniteModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return GraniteModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return GraniteConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return GPT2Tokenizer.class;
    }
}
