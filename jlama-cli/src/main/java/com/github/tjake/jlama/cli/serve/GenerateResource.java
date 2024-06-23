/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.cli.serve;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.tokenizer.PromptSupport;
import java.io.IOException;
import java.util.UUID;
import javax.validation.constraints.NotNull;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.StreamingOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

        if (params.prompt == null) {
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity("prompt is required")
                    .build();
        }

        String prompt = params.prompt;
        if (model.getTokenizer().promptSupport().isPresent()) {
            PromptSupport.Builder builder =
                    model.getTokenizer().promptSupport().get().newBuilder();
            builder.addUserMessage(prompt);
            prompt = builder.build();
        }

        final String finalPrompt = prompt;

        StreamingOutput so =
                os -> model.generate(sessionId, finalPrompt, 0.7f, Integer.MAX_VALUE, false, (s, timing) -> {
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
