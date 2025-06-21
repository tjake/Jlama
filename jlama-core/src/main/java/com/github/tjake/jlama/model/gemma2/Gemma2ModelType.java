package com.github.tjake.jlama.model.gemma2;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.gemma.GemmaTokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

public class Gemma2ModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return Gemma2Model.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return Gemma2Config.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return GemmaTokenizer.class;
    }
}
