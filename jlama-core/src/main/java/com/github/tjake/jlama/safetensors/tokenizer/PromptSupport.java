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
package com.github.tjake.jlama.safetensors.tokenizer;

import com.hubspot.jinjava.Jinjava;
import com.hubspot.jinjava.JinjavaConfig;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * This class also renders the prompt templates of the huggingface model format (using jinja templates)
 * @see <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models">Chat Templating</a>
 */
public class PromptSupport {
    // This matches the jinja config in huggingface
    private static final Jinjava jinjava = new Jinjava(JinjavaConfig.newBuilder()
            .withLstripBlocks(true)
            .withTrimBlocks(true)
            .build());

    private final TokenizerModel m;

    public PromptSupport(TokenizerModel model) {
        this.m = model;
    }

    public Builder newBuilder() {
        return new Builder(this.m);
    }

    public boolean hasPromptTemplates() {
        return !m.promptTemplates().isEmpty();
    }

    private enum PromptType {
        DEFAULT,
        TOOL,
        RAG
    }

    private enum PromptRole {
        USER,
        SYSTEM,
        ASSISTANT
    }

    static class Message {
        private final String content;
        private final PromptRole role;

        public Message(String content, PromptRole role) {
            this.content = content;
            this.role = role;
        }

        public String getContent() {
            return content;
        }

        public String getRole() {
            return role.name().toLowerCase();
        }
    }

    public static class Builder {
        private final TokenizerModel m;
        private PromptType type = PromptType.DEFAULT;
        private boolean addGenerationPrompt = true;

        private List<Message> messages = new ArrayList<>(2);

        private Builder(TokenizerModel m) {
            this.m = m;
        }

        public Builder type(PromptType type) {
            this.type = type;
            return this;
        }

        public Builder addGenerationPrompt(boolean addGenerationPrompt) {
            this.addGenerationPrompt = addGenerationPrompt;
            return this;
        }

        public Builder addUserMessage(String content) {
            messages.add(new Message(content, PromptRole.USER));
            return this;
        }

        public Builder addSystemMessage(String content) {
            messages.add(new Message(content, PromptRole.SYSTEM));
            return this;
        }

        public Builder addAssistantMessage(String content) {
            messages.add(new Message(content, PromptRole.ASSISTANT));
            return this;
        }

        public String build() {
            if (messages.isEmpty()) {
                return "";
            }

            if (m.promptTemplates().isEmpty()) {
                throw new UnsupportedOperationException("Prompt templates are not available for this model");
            }

            String template = m.promptTemplates()
                    .map(t -> t.get(type.name().toLowerCase()))
                    .orElseThrow(
                            () -> new UnsupportedOperationException("Prompt template not available for type: " + type));

            return jinjava.render(
                    template,
                    Map.of(
                            "messages",
                            messages,
                            "add_generation_prompt",
                            addGenerationPrompt,
                            "eos_token",
                            m.eosToken(),
                            "bos_token",
                            m.bosToken()));
        }
    }
}
