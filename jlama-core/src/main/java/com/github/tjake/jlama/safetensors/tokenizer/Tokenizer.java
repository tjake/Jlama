package com.github.tjake.jlama.safetensors.tokenizer;

import java.util.List;

/**
 * Tokenizer interface
 */
public interface Tokenizer {

    /**
     * Tokenize a sentence
     * @param sentence
     * @return list of token strings
     */
    List<String> tokenize(String sentence);

    /**
     * Encode a sentence into a list of token ids
     * @param sentence
     * @return list of token ids
     */
    long[] encode(String sentence);

    /**
     * Decode a token id into its string representation
     * @param id
     * @return token string
     */
    String decode(long id);

    /**
     * Decode a list of token ids into their string representation
     * @param ids list of token ids
     * @return list of token strings
     */
    String decode(long[] ids);
}
