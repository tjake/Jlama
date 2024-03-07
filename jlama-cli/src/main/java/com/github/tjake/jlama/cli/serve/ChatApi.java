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


import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import jakarta.ws.rs.Path;
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

    public Object createChatCompletion(@Context HttpServletResponse response, @Valid @NotNull CreateChatCompletionRequest createChatCompletionRequest) {
        Boolean is_streaming = createChatCompletionRequest.getStream();
        if (!is_streaming){
            // TODO: implement
            return Response.serverError().entity("only streaming is supported right now").build();
        }
        else {

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
            @Min(1) @Max(128) Integer n = createChatCompletionRequest.getN().intValue();
            try (PrintWriter writer= response.getWriter()) { // Auto-closes the sink
                new Thread(() -> {
                    try {
                        //TODO: somewhere validate that the model you're requesting is the model that's loaded, maybe support loading different models based on the request
                        model.generate(
                                sessionId,
                                model.wrapPrompt(finalPrompt, finalSystemPrompt),
                                "",  // WTF is a clean prompt?
                                temperature,
                                n,
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
                    } catch (Exception e) {
                        Thread.currentThread().interrupt();
                        e.printStackTrace();
                    }
                }).start();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return Response.ok().build();
    }
}
/*
    public Object createChatCompletion(@Context SseEventSink sseEventSink, @Valid @NotNull CreateChatCompletionRequest createChatCompletionRequest) {
        Boolean is_streaming = createChatCompletionRequest.getStream();
        if (!is_streaming) {
            // TODO: implement
            return Response.serverError().entity("only streaming is supported right now").build();
        } else {

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
            @Min(1) @Max(128) Integer n = createChatCompletionRequest.getN().intValue();
            try (SseEventSink sink = sseEventSink) { // Auto-closes the sink
                new Thread(() -> {
                    try {
                        //TODO: somewhere validate that the model you're requesting is the model that's loaded, maybe support loading different models based on the request
                        model.generate(
                                sessionId,
                                model.wrapPrompt(finalPrompt, finalSystemPrompt),
                                "",  // WTF is a clean prompt?
                                temperature,
                                n,
                                false,
                                (s, timing) -> {
                                    logger.info("'{}' took {}ms", s, timing);
                                    CreateChatCompletionStreamResponseChoicesInner inner = new CreateChatCompletionStreamResponseChoicesInner();
                                    inner.delta(new ChatCompletionStreamResponseDelta().content(s).role(ChatCompletionStreamResponseDelta.RoleEnum.ASSISTANT));
                                    CreateChatCompletionStreamResponse completionResponse = new CreateChatCompletionStreamResponse().created(0).addChoicesItem(inner);

                                    OutboundSseEvent event = sseBuilder
                                            .name("chat-completion-delta")
                                            .id(sessionId.toString())
                                            .data(completionResponse)
                                            .build();
                                    sink.send(event);
                                });
                    } catch (Exception e) {
                        Thread.currentThread().interrupt();
                        e.printStackTrace();
                    }
                }).start();
                //} catch (IOException e) {
                //    throw new RuntimeException(e);
                //}
            }
            return Response.ok().build();
        }
    }
}
    */