package com.github.tjake.jlama.cli.serve;

import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;

@Path("/embed")
@Produces(MediaType.APPLICATION_JSON)
public class EmbedResource {
    @POST
    public String embed(String text) {
        return text;
    }
}
