package com.github.tjake.jlama.cli.serve;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public class GenerateParams {
    @JsonProperty("prompt")
    public String prompt;

    @JsonProperty("temperature")
    public Float temp;
}
