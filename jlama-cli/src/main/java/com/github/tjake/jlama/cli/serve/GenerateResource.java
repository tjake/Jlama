package com.github.tjake.jlama.cli.serve;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.functions.Generator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.validation.constraints.NotNull;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.StreamingOutput;
import java.io.IOException;
import java.util.Optional;
import java.util.UUID;

@Path("/generate")
@Consumes(MediaType.APPLICATION_JSON)
public class GenerateResource {
    private static final ObjectMapper om = new ObjectMapper();
    private static final Logger logger = LoggerFactory.getLogger(GenerateResource.class);

    final Generator model;
    public GenerateResource(Generator model) {
        this.model = model;
    }

    @POST
    public Response generate(@NotNull GenerateParams params) {
        logger.debug("Sending generate request: {}", params);
        UUID sessionId = params.sessionId == null ? UUID.randomUUID() : params.sessionId;
        StreamingOutput so = os -> model.generate(sessionId, model.wrapPrompt(params.prompt, Optional.empty()), "", 0.7f, 256, false, (s, timing) -> {
            try {
                logger.info("'{}' took {}ms", s, timing);
                os.write(om.writeValueAsBytes(new GenerateResponse(s, false)));
                os.write("\n".getBytes());
                os.flush();
            } catch (IOException e) {
                logger.warn("streaming exception", e);
            }
        });

        return Response.ok(so, "application/x-ndjson").build();
    }
}
