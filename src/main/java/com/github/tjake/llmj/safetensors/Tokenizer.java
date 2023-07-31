package com.github.tjake.llmj.safetensors;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public interface Tokenizer {
    List<String> tokenize(String sentence);

    long[] encode(String sentence);

    String decode(long id);

    String decode(long[] ids);
}
