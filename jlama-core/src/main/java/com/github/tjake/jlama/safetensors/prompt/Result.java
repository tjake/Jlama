package com.github.tjake.jlama.safetensors.prompt;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.github.tjake.jlama.util.JsonSupport;

/**
 * Result
 */
@JsonPropertyOrder({Result.JSON_PROPERTY_OUTPUT})
public class Result {
    public static final String JSON_PROPERTY_OUTPUT = "output";
    private final Object output;

    private Result(Object output) {
        this.output = output;
    }

    public static Result from(Object output) {
        return new Result(output);
    }

    public Object getOutput() {
        return output;
    }

    public String toJson() {
        return JsonSupport.toJson(this);
    }
}
