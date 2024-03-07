package com.github.tjake.jlama.cli.serve;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.functions.Generator;

import jakarta.validation.constraints.NotNull;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
