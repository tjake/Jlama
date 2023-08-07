package com.github.tjake.llmj.safetensors;

import java.util.List;

public interface Tokenizer {
    List<String> tokenize(String sentence);

    long[] encode(String sentence);

    String decode(long id);

    String decode(long[] ids);
}
