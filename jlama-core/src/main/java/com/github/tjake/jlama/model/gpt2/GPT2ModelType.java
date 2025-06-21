package com.github.tjake.jlama.model.gpt2;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

public class GPT2ModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return GPT2Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return GPT2Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return GPT2Tokenizer.class;
    }
}
