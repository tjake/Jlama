package com.github.tjake.jlama.cli.serve;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.cli.serve.model.*;

import com.github.tjake.jlama.model.functions.Generator;
import io.swagger.annotations.*;
import jakarta.validation.Valid;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.Response;
import jakarta.servlet.http.HttpServletResponse;


import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import jakarta.ws.rs.Path;
import jakarta.ws.rs.core.StreamingOutput;
import jakarta.ws.rs.sse.OutboundSseEvent;
import jakarta.ws.rs.sse.Sse;
import jakarta.ws.rs.sse.SseEventSink;
import org.jboss.resteasy.plugins.providers.sse.SseImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
* Represents a collection of functions to interact with the API endpoints.
*/
@Path("/chat/completions")
@Api(description = "the chat API")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class ChatApi {

    final Generator model;
    private OutboundSseEvent.Builder sseBuilder = new SseImpl().newEventBuilder();

    public ChatApi(Generator model) {
        this.model = model;
        System.out.println("ChatApi constructor");
    }

    private static final Logger logger = LoggerFactory.getLogger(ChatApi.class);
    private static final ObjectMapper om = new ObjectMapper();

    private Sse sse; // Field to hold the Sse instance

    @Context
    public void setSse(Sse sse) {
        this.sse = sse; // Initialize the Sse instance using @Context
    }

    @POST
    @Consumes({"application/json"})
    @Produces({"application/json"})
    //@Produces({ "event/stream" })
    @ApiOperation(value = "Creates a model response for the given chat conversation.", notes = "", response = CreateChatCompletionResponse.class, authorizations = {
    }, tags = {"Chat"})
    @ApiResponses(value = {
            @ApiResponse(code = 200, message = "OK")
    })

    public Object createChatCompletion(@Context HttpServletResponse response, @Valid @NotNull CreateChatCompletionRequest createChatCompletionRequest) throws IOException {
        logger.debug("Sending generate request: {}", createChatCompletionRequest);
        UUID sessionId = UUID.randomUUID();

        String prompt = "";
        String systemPrompt = "";

        @NotNull @Size(min = 1) List<ChatCompletionRequestMessage> messages = createChatCompletionRequest.getMessages();
        for (ChatCompletionRequestMessage message : messages) {
            if (message.getRole() == ChatCompletionRole.ASSISTANT || message.getRole() == ChatCompletionRole.USER) {
                prompt += message.getContent();
            }
            if (message.getRole() == ChatCompletionRole.SYSTEM) {
                systemPrompt += message.getContent();
            }
            if (message.getRole() == ChatCompletionRole.FUNCTION) {
                return Response.serverError().entity("function role is not supported").build();
            }
            if (message.getRole() == ChatCompletionRole.TOOL) {
                return Response.serverError().entity("tool role is not supported").build();
            }
        }
        String finalPrompt = prompt;
        Optional<String> finalSystemPrompt = Optional.of(systemPrompt);

        float temperature = createChatCompletionRequest.getTemperature().floatValue();
        Integer numTokens = createChatCompletionRequest.getMaxTokens();
        if (numTokens == null || numTokens < 1) {
            numTokens = 1024;
        }
        int finalNumTokens = numTokens;

        Boolean is_streaming = createChatCompletionRequest.getStream();
        if (!is_streaming){
            //TODO: somewhere validate that the model you're requesting is the model that's loaded, maybe support loading different models based on the request
            StreamingOutput so = os -> model.generate(
                    sessionId,
                    model.wrapPrompt(finalPrompt, finalSystemPrompt),
                    "",  // WTF is a clean prompt?
                    temperature,
                    finalNumTokens,
                    false,
                    (s, timing) -> {
                        try {
                            logger.info("'{}' took {}ms", s, timing);
                            os.write(s.getBytes(StandardCharsets.UTF_8));
                            os.flush();
                        } catch (IOException e) {
                            logger.warn("streaming exception", e);
                        }
                    });
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            so.write(baos);

            String content = baos.toString(StandardCharsets.UTF_8);
            CreateChatCompletionResponseChoicesInner inner = new CreateChatCompletionResponseChoicesInner();
            ChatCompletionResponseMessage message = new ChatCompletionResponseMessage()
                    .content(content)
                    .role(ChatCompletionResponseMessage.RoleEnum.ASSISTANT);
            inner.message(message);
            CreateChatCompletionResponse chatCompletionResponse = new CreateChatCompletionResponse()
                    .created(0)
                    .addChoicesItem(inner);

            return Response.ok().entity(chatCompletionResponse).build();
        }
        else {
           try (PrintWriter writer= response.getWriter()) { // Auto-closes the sink
                //TODO: somewhere validate that the model you're requesting is the model that's loaded, maybe support loading different models based on the request
                model.generate(
                        sessionId,
                        model.wrapPrompt(finalPrompt, finalSystemPrompt),
                        "",  // WTF is a clean prompt?
                        temperature,
                        finalNumTokens,
                        false,
                        (s, timing) -> {
                            if (!s.equals("")) {
                                logger.info("'{}' took {}ms", s, timing);
                                CreateChatCompletionStreamResponseChoicesInner inner = new CreateChatCompletionStreamResponseChoicesInner();
                                inner.delta(new ChatCompletionStreamResponseDelta().content(s).role(ChatCompletionStreamResponseDelta.RoleEnum.ASSISTANT));
                                CreateChatCompletionStreamResponse completionResponse = new CreateChatCompletionStreamResponse().created(0).addChoicesItem(inner);

                                StringBuilder sb = new StringBuilder();
                                sb.append("data: ");
                                try {
                                    sb.append(om.writeValueAsString(completionResponse));
                                    sb.append("\n\n");
                                    writer.write(sb.toString());
                                    writer.flush();
                                } catch (JsonProcessingException e) {
                                    throw new RuntimeException(e);
                                }
                            }
                        });
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return Response.ok().build();
    }
}