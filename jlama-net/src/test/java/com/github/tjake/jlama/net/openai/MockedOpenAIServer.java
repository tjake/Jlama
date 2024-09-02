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
package com.github.tjake.jlama.net.openai;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.net.JlamaServiceTest;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import java.io.File;
import java.io.IOException;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@SpringBootApplication
@SpringBootConfiguration
@Configuration
public class MockedOpenAIServer {
    private final JlamaServiceTest.MockConfig modelConfig = new JlamaServiceTest.MockConfig(128, 4096, 8192, 16, 12, 1e5f);

    public static void main(String[] args) {
        SpringApplication.run(MockedOpenAIServer.class, args);
    }

    @Bean
    public AbstractModel getModel() throws IOException {
        String model = "tjake/TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String workingDirectory = "./models";

        // Downloads the model or just returns the local path if it's already downloaded
        File localModelPath = SafeTensorSupport.maybeDownloadModel(workingDirectory, model);

        // Loads the model
        return ModelSupport.loadModel(localModelPath, DType.F32, DType.I8);
    }
}
