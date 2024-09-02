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
package com.github.tjake.jlama.net.openai;

import io.github.stefanbratanov.jvm.openai.*;
import org.json.JSONException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpHeaders;
import org.springframework.test.context.junit.jupiter.SpringExtension;

@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = MockedOpenAIServer.class, webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, useMainMethod = SpringBootTest.UseMainMethod.ALWAYS)
public class ChatApiTest {
    @LocalServerPort
    private int port;

    TestRestTemplate restTemplate = new TestRestTemplate();

    HttpHeaders headers = new HttpHeaders();

    @Test
    public void testChatCompletion() throws JSONException {

        OpenAI openAI = OpenAI.newBuilder("Fake key").baseUrl("http://localhost:" + port).build();

        ChatClient client = openAI.chatClient();

        CreateChatCompletionRequest request = CreateChatCompletionRequest.newBuilder()
            .model(OpenAIModel.GPT_3_5_TURBO)
            .stream(false)
            .temperature(0.0f)
            .message(ChatMessage.userMessage("Who won the world series in 2020?"))
            .build();

        ChatCompletion response = client.createChatCompletion(request);

        System.err.println(response);

        // JSONAssert.assertEquals(null, response.getBody(), false);
    }

    @Test
    public void testStreamingChatCompletion() throws JSONException {

        OpenAI openAI = OpenAI.newBuilder("Fake key").baseUrl("http://localhost:" + port).build();

        ChatClient client = openAI.chatClient();

        CreateChatCompletionRequest request = CreateChatCompletionRequest.newBuilder()
            .model(OpenAIModel.GPT_3_5_TURBO)
            .stream(true)
            .temperature(0.0f)
            .message(ChatMessage.userMessage("Who won the world series in 2020?"))
            .build();

        client.streamChatCompletion(request).forEach(System.err::println);

        // JSONAssert.assertEquals(null, response.getBody(), false);
    }

    private String createURLWithPort(String uri) {
        return "http://localhost:" + port + uri;
    }
}
