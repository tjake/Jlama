package com.github.tjake.jlama.net.openai;


import io.github.stefanbratanov.jvm.openai.*;
import org.json.JSONException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.skyscreamer.jsonassert.JSONAssert;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.math.BigDecimal;
import java.util.List;

@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = MockedOpenAIServer.class, webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, useMainMethod = SpringBootTest.UseMainMethod.ALWAYS)
public class ChatApiTest {
    @LocalServerPort
    private int port;

    TestRestTemplate restTemplate = new TestRestTemplate();

    HttpHeaders headers = new HttpHeaders();

    @Test
    public void testChatCompletion() throws JSONException {

        OpenAI openAI = OpenAI.newBuilder("Fake key")
                .baseUrl("http://localhost:" + port)
                .build();

        ChatClient client = openAI.chatClient();

        CreateChatCompletionRequest request = CreateChatCompletionRequest.newBuilder()
                .model(OpenAIModel.GPT_3_5_TURBO)
                .stream(false)
                .temperature(0.0f)
                .message(ChatMessage.userMessage("Who won the world series in 2020?"))
                .build();

        ChatCompletion response = client.createChatCompletion(request);

        System.err.println(response);

        //JSONAssert.assertEquals(null, response.getBody(), false);
    }

    @Test
    public void testStreamingChatCompletion() throws JSONException {

        OpenAI openAI = OpenAI.newBuilder("Fake key")
                .baseUrl("http://localhost:" + port)
                .build();

        ChatClient client = openAI.chatClient();

        CreateChatCompletionRequest request = CreateChatCompletionRequest.newBuilder()
                .model(OpenAIModel.GPT_3_5_TURBO)
                .stream(true)
                .temperature(0.0f)
                .message(ChatMessage.userMessage("Who won the world series in 2020?"))
                .build();

        client.streamChatCompletion(request).forEach(System.err::println);


        //JSONAssert.assertEquals(null, response.getBody(), false);
    }

    private String createURLWithPort(String uri) {
        return "http://localhost:" + port + uri;
    }
}
