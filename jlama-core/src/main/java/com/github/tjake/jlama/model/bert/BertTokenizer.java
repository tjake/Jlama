package com.github.tjake.jlama.model.bert;

import com.github.tjake.jlama.safetensors.tokenizer.WordPieceTokenizer;

import java.nio.file.Path;

public class BertTokenizer extends WordPieceTokenizer {

    public BertTokenizer(Path modelRoot) {
        super(modelRoot);
    }
}
