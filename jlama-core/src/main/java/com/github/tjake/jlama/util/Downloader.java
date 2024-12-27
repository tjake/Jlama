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

import com.github.tjake.jlama.safetensors.SafeTensorSupport;

import java.io.File;
import java.io.IOException;
import java.util.Optional;

public class Downloader {

    private static final String AUTH_TOKEN_ENV_VAR = "HF_TOKEN";
    private static final String AUTH_TOKEN_PROP = "huggingface.auth.token";
    private final String modelDir;
    private final String modelOwner;
    private final String modelName;
    private boolean downloadWeights = true;
    private String branch;
    private String authToken;
    private ProgressReporter progressReporter;

    public Downloader(String modelDir, String model) {

        String[] parts = model.split("/");
        if (parts.length == 0 || parts.length > 2) {
            throw new IllegalArgumentException("Model must be in the form owner/name");
        }

        String owner;
        String name;

        if (parts.length == 1) {
            owner = null;
            name = model;
        } else {
            owner = parts[0];
            name = parts[1];
        }

        this.modelDir = modelDir;
        this.modelOwner = owner;
        this.modelName = name;

    }

    public Downloader skipWeights() {
        this.downloadWeights = false;
        return this;
    }

    public Downloader withBranch(String branch) {
        this.branch = branch;
        return this;
    }

    public Downloader withAuthToken(String token) {
        this.authToken = token;
        return this;
    }

    public Downloader withProgressReporter(ProgressReporter progressReporter) {
        this.progressReporter = progressReporter;
        return this;
    }

    public File huggingFaceModel() throws IOException {
        return SafeTensorSupport.maybeDownloadModel(
            this.modelDir,
            Optional.of(this.modelOwner),
            this.modelName,
            this.downloadWeights,
            Optional.ofNullable(this.branch),
            Optional.ofNullable(getAuthToken()),
            Optional.ofNullable(this.progressReporter)
        );
    }

    private String getAuthToken() {
        String token = System.getenv(AUTH_TOKEN_ENV_VAR);
        if (token == null) {
            token = System.getProperty(AUTH_TOKEN_PROP);
            if (token == null) {
                token = this.authToken;
            }
        }

        return token;
    }

}
