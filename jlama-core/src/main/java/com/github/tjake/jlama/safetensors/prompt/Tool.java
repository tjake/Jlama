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

import com.fasterxml.jackson.annotation.JsonPropertyOrder;

/**
 * Tool
 */
@JsonPropertyOrder({ Tool.JSON_PROPERTY_TYPE, Tool.JSON_PROPERTY_FUNCTION })
public class Tool {
    public static final String JSON_PROPERTY_TYPE = "type";
    private final String type = "function";

    public static final String JSON_PROPERTY_FUNCTION = "function";
    private final Function function;

    private Tool(Function function) {
        this.function = function;
    }

    public static Tool from(Function function) {
        return new Tool(function);
    }

    public Function getFunction() {
        return function;
    }

    public String getType() {
        return type;
    }
}
