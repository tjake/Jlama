package com.github.tjake.jlama.safetensors.prompt;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.*;

/**
 * Parameters
 */
@JsonPropertyOrder({
        Parameters.JSON_PROPERTY_TYPE,
        Parameters.JSON_PROPERTY_PROPERTIES,
        Parameters.JSON_PROPERTY_REQUIRED})
public class Parameters {

    public static final String JSON_PROPERTY_TYPE = "type";
    private final String type = "object";

    public static final String JSON_PROPERTY_PROPERTIES = "properties";
    private final Map<String, Map<String, Object>> properties;

    public static final String JSON_PROPERTY_REQUIRED = "required";
    private List<String> required;

    public Parameters(Map<String, Map<String, Object>> properties, List<String> required) {
        this.properties = properties;
        this.required = required;
    }

    public Parameters(Map<String, Map<String, Object>> properties) {
        this.properties = properties;
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

    public Map<String, Map<String, Object>> getProperties() {
        return properties;
    }

    public List<String> getRequired() {
        return required;
    }
}