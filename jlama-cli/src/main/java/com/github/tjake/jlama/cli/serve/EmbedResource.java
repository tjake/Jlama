package com.github.tjake.jlama.cli.serve;

import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

@Path("/embed")
@Produces(MediaType.APPLICATION_JSON)
public class EmbedResource {
    @POST
    public String embed(String text) {
        return text;
    }
}
