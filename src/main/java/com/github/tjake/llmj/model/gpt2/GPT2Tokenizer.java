package com.github.tjake.llmj.model.gpt2;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.github.tjake.llmj.safetensors.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class GPT2Tokenizer implements Tokenizer {
    private final HuggingFaceTokenizer tokenizer;

    public GPT2Tokenizer(Path tokenizerJson) throws IOException {
        tokenizer = HuggingFaceTokenizer.newInstance(tokenizerJson);
    }

    public List<String> tokenize(String sentence) {
        return tokenizer.tokenize(sentence);
    }

    public long[] encode(String sentence)
    {
        return tokenizer.encode(sentence).getIds();
    }

    @Override
    public String decode(long id) {
        return decode(new long[]{id});
    }

    public String decode(long[] ids)
    {
        return tokenizer.decode(ids);
    }
}
