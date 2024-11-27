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
package com.github.tjake.jlama.util;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Helper class for Jackson JSON support
 */
public class JsonSupport {
       public static final ObjectMapper om = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
        .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
        .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, false)
        .enable(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS);

    public static String toJson(Object o) {
        try {
            return om.writer(JlamaPrettyPrinter.INSTANCE).writeValueAsString(o);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static List<String> extractJsonFromString(String s) {
        JsonFactory factory = new JsonFactory();
        String text = s;
        List<String> jsons = new ArrayList<>();

        while (text != null && !text.isEmpty()) {
            boolean found = false;
            for (int i = 0; i < text.length(); i++) {
                if (text.charAt(i) == '{' || text.charAt(i) == '[') {
                    String possibleJson = text.substring(i);
                    try (JsonParser parser = factory.createParser(new StringReader(possibleJson))) {
                        JsonToken token = parser.nextToken();
                        if (token != null) {
                            parser.skipChildren();
                            int endIndex = i + (int) parser.getTokenLocation().getCharOffset();
                            int extra = 0;
                            while (endIndex + extra < text.length() && (text.charAt(endIndex + extra) == '}' || text.charAt(endIndex + extra) == ']')) {
                                extra++;
                            }
                            String jsonString = text.substring(i, endIndex + extra);
                            jsons.add(jsonString);
                            found = true;
                            text = text.substring(endIndex + extra);
                            // Parse the valid JSON string as needed
                            break;
                        }
                    } catch (Exception e) {
                        // Not a valid JSON at this position
                    }
                }
            }

            if (!found) {
                break;
            }
        }

        return jsons;
    }

    public static class JlamaPrettyPrinter extends DefaultPrettyPrinter {

        public static final JlamaPrettyPrinter INSTANCE = new JlamaPrettyPrinter();

        @Override
        public DefaultPrettyPrinter createInstance() {
            return INSTANCE;
        }

        private JlamaPrettyPrinter() {
            _objectIndenter = FixedSpaceIndenter.instance;
            _spacesInObjectEntries = false;
        }

        @Override
        public void beforeArrayValues(JsonGenerator jg) {}

        @Override
        public void writeEndArray(JsonGenerator jg, int nrOfValues) throws IOException {
            if (!this._arrayIndenter.isInline()) {
                --this._nesting;
            }
            jg.writeRaw(']');
        }

        @Override
        public void writeObjectFieldValueSeparator(JsonGenerator jg) throws IOException {
            jg.writeRaw(": ");
        }

        @Override
        public void beforeObjectEntries(JsonGenerator jg) {}

        @Override
        public void writeEndObject(JsonGenerator jg, int nrOfEntries) throws IOException {
            if (!this._objectIndenter.isInline()) {
                --this._nesting;
            }
            jg.writeRaw("}");
        }
    }
}
