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
import com.google.common.collect.ImmutableMap;
import com.hubspot.jinjava.objects.collections.PyMap;
import java.util.*;

/**
 * Parameters
 */
@JsonPropertyOrder({ Parameters.JSON_PROPERTY_TYPE, Parameters.JSON_PROPERTY_PROPERTIES, Parameters.JSON_PROPERTY_REQUIRED })
public class Parameters extends PyMap {

    public static final String JSON_PROPERTY_TYPE = "type";
    private final String type = "object";

    public static final String JSON_PROPERTY_PROPERTIES = "properties";

    public static final String JSON_PROPERTY_REQUIRED = "required";
    private List<String> required;

    public Parameters(Map<String, Map<String, Object>> properties, List<String> required) {
        super(
            ImmutableMap.<String, Object>builder()
                .put(JSON_PROPERTY_TYPE, "object")
                .put(JSON_PROPERTY_PROPERTIES, properties)
                .put(JSON_PROPERTY_REQUIRED, required)
                .build()
        );
        this.required = required;
    }

    public Parameters(Map<String, Map<String, Object>> properties) {
        super(ImmutableMap.<String, Object>builder().put(JSON_PROPERTY_TYPE, "object").put(JSON_PROPERTY_PROPERTIES, properties).build());
        this.required = null;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Map<String, Map<String, Object>> properties = new LinkedHashMap<>();
        private Set<String> required = new LinkedHashSet<>();

        public Builder addProperty(String name, String type, String description, boolean required) {
            Map<String, Object> properties = new LinkedHashMap<>();
            properties.put("type", type);
            properties.put("description", description);
            return addProperty(name, properties, required);
        }

        public Builder addProperty(String name, Map<String, Object> properties, boolean required) {
            this.properties.put(name, properties);
            if (required) {
                this.required.add(name);
            }
            return this;
        }

        public Parameters build() {
            return new Parameters(properties, new ArrayList<>(required));
        }
    }

    public String getType() {
        return this.type;
    }

    public Map<String, Object> getProperties() {
        return this;
    }

    public List<String> getRequired() {
        return required;
    }
}
