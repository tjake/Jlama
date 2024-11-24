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
package com.github.tjake.jlama.safetensors.prompt;


import com.fasterxml.jackson.annotation.*;

import java.util.Map;
import java.util.Objects;

@JsonPropertyOrder({ ToolResult.JSON_PROPERTY_TOOL_NAME, ToolResult.JSON_PROPERTY_TOOL_ID })
public class ToolCall {
    @JsonProperty("name")
    private final String name;

    @JsonProperty("id")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String id;

    @JsonProperty("parameters")
    private final Map<String, Object> parameters;

    @JsonCreator
    public ToolCall(
        @JsonProperty("name") String name,
        @JsonAlias({ "arguments" }) @JsonProperty("parameters") Map<String, Object> parameters
    ) {
        this.name = name;
        this.parameters = parameters;
    }

    public ToolCall(String name, String id, Map<String, Object> parameters) {
        this.name = name;
        this.id = id;
        this.parameters = parameters;
    }

    public String getName() {
        return name;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ToolCall toolCall = (ToolCall) o;
        return Objects.equals(name, toolCall.name) && Objects.equals(parameters, toolCall.parameters);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, parameters);
    }
}
