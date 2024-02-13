package com.github.tjake.jlama.model.mistral;

import com.github.tjake.jlama.model.llama.LlamaModel;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;

import java.util.Optional;

public class MistralModel extends LlamaModel {

    public MistralModel(Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    public MistralModel(InferenceType inferenceType, Config config, WeightLoader weights, Tokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType);
    }

    @Override
    public String wrapPrompt(String prompt, Optional<String> systemPrompt)
    {
        StringBuilder b = new StringBuilder();

        if (systemPrompt.isPresent()) {
            b.append(systemPrompt.get())
                    .append("\n\n");
        }

        b.append("[INST] ").append(prompt).append(" [/INST]");

        return b.toString();
    }
}
