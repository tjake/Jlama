package com.github.tjake.jlama.model.bert;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

public class BertModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return BertModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return BertConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return BertTokenizer.class;
    }
}
