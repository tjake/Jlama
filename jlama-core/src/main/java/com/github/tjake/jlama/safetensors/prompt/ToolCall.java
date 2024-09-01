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

import static com.github.tjake.jlama.util.JsonSupport.om;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.type.ArrayType;
import java.util.Map;

public class ToolCall {
    public static final ArrayType toolCallListTypeReference =
            om.getTypeFactory().constructArrayType(ToolCall.class);

    @JsonProperty("name")
    private final String name;

    @JsonProperty("parameters")
    private final Map<String, Object> parameters;

    @JsonCreator
    public ToolCall(
            @JsonProperty("name") String name,
            @JsonAlias({"arguments"}) @JsonProperty("parameters") Map<String, Object> parameters) {
        this.name = name;
        this.parameters = parameters;
    }

    public String getName() {
        return name;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }
}
