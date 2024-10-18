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

import com.google.common.io.CountingInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.zip.GZIPInputStream;

public class HttpSupport {
    public static final Logger logger = LoggerFactory.getLogger(HttpSupport.class);

    public static Pair<InputStream, Long> getResponse(
        String urlString,
        Optional<String> optionalAuthHeader,
        Optional<Pair<Long, Long>> optionalByteRange
    ) throws IOException {
        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();

        // Set the request method
        connection.setRequestMethod("GET");
        connection.setRequestProperty("Accept-Encoding", "gzip");

        // Set the request header
        optionalAuthHeader.ifPresent(authHeader -> connection.setRequestProperty("Authorization", "Bearer " + authHeader));
        optionalByteRange.ifPresent(byteRange -> connection.setRequestProperty("Range", "bytes=" + byteRange.left + "-" + byteRange.right));

        // Get the response code
        int responseCode = connection.getResponseCode();

        if (responseCode == HttpURLConnection.HTTP_OK || responseCode == HttpURLConnection.HTTP_PARTIAL) {
            // If the response code is 200 (HTTP_OK), return the input stream

            String encoding = connection.getContentEncoding();
            InputStream inputStream;
            if (encoding != null && encoding.equals("gzip")) {
                inputStream = new GZIPInputStream(connection.getInputStream());
            } else {
                inputStream = connection.getInputStream();
            }

            return Pair.of(inputStream, connection.getContentLengthLong());
        } else {
            // If the response code is not 200/206, throw an IOException
            throw new IOException("HTTP response code: " + responseCode + " for URL: " + urlString);
        }
    }

    public static String readInputStream(InputStream inStream) throws IOException {
        if (inStream == null) return null;

        BufferedReader inReader = new BufferedReader(new InputStreamReader(inStream));
        StringBuilder stringBuilder = new StringBuilder();

        String currLine;
        while ((currLine = inReader.readLine()) != null) {
            stringBuilder.append(currLine);
            stringBuilder.append(System.lineSeparator());
        }

        return stringBuilder.toString();
    }

    public static void downloadFile(
        String hfModel,
        String currFile,
        Optional<String> optionalBranch,
        Optional<String> optionalAuthHeader,
        Optional<Pair<Long, Long>> optionalByteRange,
        Path outputPath,
        Optional<ProgressReporter> optionalProgressConsumer
    ) throws IOException {

        Pair<InputStream, Long> stream = getResponse(
            "https://huggingface.co/" + hfModel + "/resolve/" + optionalBranch.orElse("main") + "/" + currFile,
            optionalAuthHeader,
            optionalByteRange
        );

        CountingInputStream inStream = new CountingInputStream(stream.left);

        long totalBytes = stream.right;

        if (outputPath.toFile().exists() && outputPath.toFile().length() == totalBytes) {
            logger.debug("File already exists: {}", outputPath);
            return;
        }

        if (optionalProgressConsumer.isEmpty()) logger.info("Downloading file: {}", outputPath);

        optionalProgressConsumer.ifPresent(p -> p.update(currFile, 0L, totalBytes));

        CompletableFuture<Long> result = CompletableFuture.supplyAsync(() -> {
            try {
                return Files.copy(inStream, outputPath, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        optionalProgressConsumer.ifPresent(p -> {
            while (!result.isDone()) {
                p.update(currFile, inStream.getCount(), totalBytes);
            }

            if (result.isCompletedExceptionally()) p.update(currFile, inStream.getCount(), totalBytes);
            else p.update(currFile, totalBytes, totalBytes);
        });

        try {
            result.get();
        } catch (Throwable e) {
            throw new IOException("Failed to download file: " + currFile, e);
        }

        if (optionalProgressConsumer.isEmpty() && !result.isCompletedExceptionally()) logger.info("Downloaded file: {}", outputPath);
    }
}
