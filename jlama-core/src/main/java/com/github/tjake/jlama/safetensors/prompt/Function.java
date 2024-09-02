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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.hubspot.jinjava.objects.collections.PyMap;
import java.util.Map;

/**
 * FunctionObject
 */
@JsonPropertyOrder({ Function.JSON_PROPERTY_NAME, Function.JSON_PROPERTY_DESCRIPTION, Function.JSON_PROPERTY_PARAMETERS })
public class Function extends PyMap {
    public static final String JSON_PROPERTY_NAME = "name";
    private final String name;

    public static final String JSON_PROPERTY_DESCRIPTION = "description";
    private final String description;

    public static final String JSON_PROPERTY_PARAMETERS = "parameters";
    private final Parameters parameters;

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String name;
        private String description = "";

        private Parameters.Builder parameters = Parameters.builder();

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder addParameter(String name, String type, String description, boolean required) {
            this.parameters.addProperty(name, type, description, required);
            return this;
        }

        public Builder addParameter(String name, Map<String, Object> properties, boolean required) {
            this.parameters.addProperty(name, properties, required);
            return this;
        }

        public Function build() {
            Preconditions.checkNotNull(name, "name is required");

            return new Function(name, description, parameters.build());
        }
    }

    private Function(String name, String description, Parameters parameters) {
        super(
            ImmutableMap.<String, Object>builder()
                .put(JSON_PROPERTY_NAME, name)
                .put(JSON_PROPERTY_DESCRIPTION, description)
                .put(JSON_PROPERTY_PARAMETERS, parameters)
                .build()
        );
        this.name = name;
        this.description = description;
        this.parameters = parameters;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public Parameters getParameters() {
        return parameters;
    }
}
