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

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Helper class for Jackson JSON support
 */
public class JsonSupport {
    private static final String JSON_REGEX =
            "\\{\\s*\"name\"\\s*:\\s*\"[^\"]+\"\\s*(,\\s*\"[^\"]+\"\\s*:\\s*(\"[^\"]*\"|\\d+|true|false|null|\\{[^{}]*\\}|\\[[^\\[\\]]*\\]))*\\s*\\}";
    private static final Pattern JSON_PATTERN = Pattern.compile(JSON_REGEX);
    public static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
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
        Matcher matcher = JSON_PATTERN.matcher(s);
        List<String> extractedJson = new ArrayList<>();
        while (matcher.find()) {
            String json = matcher.group();
            extractedJson.add(json);
        }

        return extractedJson;
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
