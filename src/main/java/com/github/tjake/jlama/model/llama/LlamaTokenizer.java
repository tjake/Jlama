package com.github.tjake.jlama.model.llama;

import ai.djl.sentencepiece.SpTextEmbedding;
import ai.djl.sentencepiece.SpTokenizer;
import ai.djl.sentencepiece.SpVocabulary;
import ai.djl.util.passthrough.PassthroughNDManager;
import com.github.tjake.jlama.safetensors.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class LlamaTokenizer implements Tokenizer {

    static final String SPIECE_UNDERLINE = "▁";

    private SpTokenizer tokenizer;
    private SpTextEmbedding embedding;
    private SpVocabulary vocab;


    public LlamaTokenizer(Path modelRoot) throws IOException {
        this.tokenizer = new SpTokenizer(modelRoot, "tokenizer.model");
        this.embedding = SpTextEmbedding.from(tokenizer);
        this.vocab = SpVocabulary.from(tokenizer);
    }

    @Override
    public List<String> tokenize(String sentence) {
        return tokenizer.tokenize(sentence);
    }

    @Override
    public long[] encode(String sentence) {
        return embedding.preprocessTextToEmbed(List.of(preNormalize(sentence)));
    }

    @Override
    public String decode(long[] ids) {
        //Wow what a messed up api
        return tokenizer.buildSentence(embedding.unembedText(PassthroughNDManager.INSTANCE.create(ids)));
    }

    public String decode(long id) {
        //Handle ascii codes (shifted by 3 in vocab)
        if (id >= 3 && id < 259)
            return new String(new char[]{(char)(id-3)});

        return postNormalize(vocab.getToken(id));
    }

    private String preNormalize(String sentence) {
        return sentence.replace(" ", SPIECE_UNDERLINE);
    }

    private String postNormalize(String sentence) {
        sentence = sentence.replaceAll("</?s>", "");
        sentence = sentence.replaceAll(SPIECE_UNDERLINE, " ");
        return sentence;
    }
}
