package com.github.tjake.jlama.model.bert;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.github.tjake.jlama.safetensors.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class BertTokenizer implements Tokenizer {
    private final HuggingFaceTokenizer tokenizer;

    public BertTokenizer(Path tokenizerJson) throws IOException {
        tokenizer = HuggingFaceTokenizer.newInstance(tokenizerJson);
    }

    public List<String> tokenize(String sentence) {
        return tokenizer.tokenize(sentence);
    }

    public long[] encode(String sentence) {
        return tokenizer.encode(sentence).getIds();
    }

    @Override
    public String decode(long id) {
        return decode(new long[]{id});
    }

    public String decode(long[] ids) {
        return tokenizer.decode(ids);
    }
}
