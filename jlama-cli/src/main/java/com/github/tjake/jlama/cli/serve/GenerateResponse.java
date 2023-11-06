package com.github.tjake.jlama.cli.serve;

import com.fasterxml.jackson.annotation.JsonProperty;

public class GenerateResponse {

    @JsonProperty("response")
    public final String response;

    @JsonProperty("done")
    public final boolean done;

    public GenerateResponse(String response, boolean done) {
        this.response = response;
        this.done = done;
    }
}
