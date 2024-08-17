package com.github.tjake.jlama.safetensors.prompt;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;


/**
 * Tool
 */
@JsonPropertyOrder({
        Tool.JSON_PROPERTY_TYPE,
        Tool.JSON_PROPERTY_FUNCTION})
public class Tool {
    public static final String JSON_PROPERTY_TYPE = "type";
    private final String type = "function";

    public static final String JSON_PROPERTY_FUNCTION = "function";
    private final Function function;

    private Tool(Function function) {
        this.function = function;
    }

    public static Tool from(Function function) {
        return new Tool(function);
    }

    public Function getFunction() {
        return function;
    }

    public String getType() {
        return type;
    }
}
