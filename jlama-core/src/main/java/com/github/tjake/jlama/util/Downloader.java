package com.github.tjake.jlama.util;

import com.github.tjake.jlama.safetensors.SafeTensorSupport;

import java.io.File;
import java.io.IOException;
import java.util.Optional;

public class Downloader {

    private final String modelDir;
    private final String modelOwner;
    private final String modelName;
    private boolean downloadWeights = true;
    private String branch;
    private String authToken;
    private ProgressReporter progressReporter;

    public Downloader(String modelDir,
                      String model) {

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

    public Downloader notDownloadWieghts() {
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
        return SafeTensorSupport.maybeDownloadModel(this.modelDir, Optional.of(this.modelOwner), this.modelName,
                this.downloadWeights, Optional.ofNullable(this.branch),
                Optional.ofNullable(this.authToken), Optional.ofNullable(this.progressReporter));
    }

}
