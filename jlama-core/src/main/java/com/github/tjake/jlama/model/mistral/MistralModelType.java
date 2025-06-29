/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.model.mistral;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.llama.LlamaTokenizer;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

public class MistralModelType implements ModelSupport.ModelType {
    @Override
    public Class<? extends AbstractModel> getModelClass() {
        return MistralModel.class;
    }

    @Override
    public Class<? extends Config> getConfigClass() {
        return MistralConfig.class;
    }

    @Override
    public Class<? extends Tokenizer> getTokenizerClass() {
        return LlamaTokenizer.class;
    }
}
