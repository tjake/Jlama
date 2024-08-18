package com.github.tjake.jlama.safetensors.prompt;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public class ToolCall {
    private final String name;
    private final Map<String, Object> parameters;

    @JsonCreator
    public ToolCall(@JsonProperty("name") String name, @JsonProperty("parameters") Map<String, Object> parameters) {
        this.name = name;
        this.parameters = parameters;
    }

    public String getName() {
        return name;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }
}
