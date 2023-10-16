package com.github.tjake.jlama.model;

import java.io.File;

import com.github.tjake.jlama.model.bert.BertConfig;
import com.github.tjake.jlama.model.bert.BertModel;
import com.github.tjake.jlama.model.bert.BertTokenizer;
import com.github.tjake.jlama.model.gpt2.GPT2Config;
import com.github.tjake.jlama.model.gpt2.GPT2Model;
import com.github.tjake.jlama.model.gpt2.GPT2Tokenizer;
import com.github.tjake.jlama.model.llama.LlamaConfig;
import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.Tokenizer;

public class ModelSupport {

    public enum ModelType {
        LLAMA(LlamaModel.class, LlamaConfig.class, LlamaTokenizer.class),
        GPT2(GPT2Model.class, GPT2Config.class, GPT2Tokenizer.class),
        BERT(BertModel.class, BertConfig.class, BertTokenizer.class);

        public final Class<? extends AbstractModel> modelClass;
        public final  Class<? extends Config> configClass;
        public final Class<? extends Tokenizer> tokenizerClass;

        ModelType(Class<? extends AbstractModel> modelClass,
             Class<? extends Config> configClass,
             Class<? extends Tokenizer> tokenizerClass) {

            this.modelClass = modelClass;
            this.configClass = configClass;
            this.tokenizerClass = tokenizerClass;
        }
    }
}
