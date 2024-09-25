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
package com.github.tjake.jlama.cli.commands;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

import java.nio.file.Path;
import java.util.Optional;

import com.github.tjake.jlama.model.functions.Generator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import picocli.CommandLine;

@CommandLine.Command(name = "restapi", description = "Starts a openai compatible rest api for interacting with this model", abbreviateSynopsis = true)
@SpringBootApplication(scanBasePackages = { "com.github.tjake.jlama.net.openai", "com.github.tjake.jlama.cli.commands" })
@SpringBootConfiguration
@Configuration
public class ApiServiceCommand extends BaseCommand implements WebMvcConfigurer {
    private static final Logger logger = LoggerFactory.getLogger(ApiServiceCommand.class);

    @CommandLine.Option(names = {
        "--port" }, paramLabel = "ARG", description = "http port (default: ${DEFAULT-VALUE})", defaultValue = "8080")
    int port = 8080;

    protected static volatile Generator m;

    @Bean
    public Generator getModelBean() {
        return m;
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/ui/**").addResourceLocations("classpath:/static/ui/");
    }

    @Override
    public void run() {
        try {
            Path modelPath = SimpleBaseCommand.getModel(
                modelName,
                modelDirectory,
                downloadSection.autoDownload,
                downloadSection.branch,
                downloadSection.authToken
            );

            m = loadModel(
                modelPath.toFile(),
                workingDirectory,
                advancedSection.workingMemoryType,
                advancedSection.workingQuantizationType,
                Optional.ofNullable(advancedSection.modelQuantization),
                Optional.ofNullable(advancedSection.threadCount)
            );

            System.out.println("Chat UI: http://localhost:" + port);
            System.out.println("OpenAI Chat API: http://localhost:" + port + "/chat/completions");

            new SpringApplicationBuilder(ApiServiceCommand.class).lazyInitialization(true)
                .properties("server.port", "" + port, "logging.level.org.springframework.web", "info")
                .build()
                .run();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
