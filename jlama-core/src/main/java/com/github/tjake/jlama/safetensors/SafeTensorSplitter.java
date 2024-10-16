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
package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.util.*;

import static com.github.tjake.jlama.util.JsonSupport.om;

/** Helper class to split a large model into pieces **/
public class SafeTensorSplitter {

    // Limit chunk size to 20G
    static long MAX_CHUNK_SIZE = 20L << 30;

    static String getChunkFile(TensorInfo info, long fileSize) {
        // Map Tensor to a chunk based on its location in the model
        long fileChunk = Math.floorDiv(info.dataOffsets[1], MAX_CHUNK_SIZE);
        long totalChunks = Math.floorDiv(fileSize, MAX_CHUNK_SIZE);
        return String.format("model-%05d-of-%05d.safetensor", fileChunk, totalChunks);
    }

    public static void main(String[] args) {
        if (args.length == 0) throw new IllegalArgumentException("Missing model name");

        String modelDir = args[0];

        if (!new File(modelDir).isDirectory()) throw new IllegalArgumentException("Not a directory");

        if (Paths.get(modelDir, SafeTensorIndex.MODEL_INDEX_JSON).toFile().exists()) throw new IllegalArgumentException("Already split");

        if (!Paths.get(modelDir, SafeTensorIndex.SINGLE_MODEL_NAME).toFile().exists()) throw new IllegalArgumentException(
            "Missing model file"
        );

        WeightLoader wl = SafeTensorSupport.loadWeights(new File(modelDir));

        try {

            Map<String, TensorInfo> info = wl.tensorInfoMap();

            // First split the metadata into N chunks and adjust the offsets
            Map<String, String> tensorIndex = new LinkedHashMap<>();
            Map<String, Pair<RandomAccessFile, FileChannel>> chunkFiles = new HashMap<>();

            Map<String, Map<String, TensorInfo>> tensorsInChunk = new LinkedHashMap<>();

            for (Map.Entry<String, TensorInfo> entry : info.entrySet()) {
                TensorInfo tensorInfo = entry.getValue();
                String name = entry.getKey();

                String chunkName = getChunkFile(tensorInfo, new File(modelDir, SafeTensorIndex.SINGLE_MODEL_NAME).length());
                tensorIndex.put(name, chunkName);

                Pair<RandomAccessFile, FileChannel> chunkFile = chunkFiles.computeIfAbsent(chunkName, n -> {
                    try {
                        File tmp = File.createTempFile("jlama", "chunk");
                        tmp.deleteOnExit();
                        RandomAccessFile r = new RandomAccessFile(tmp, "rw");
                        FileChannel ch = r.getChannel();

                        return Pair.of(r, ch);

                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });

                AbstractTensor t = wl.load(name);
                FileChannel ch = chunkFile.right;
                TensorInfo newInfo = t.save(ch);
                System.out.println(
                    "Wrote " + name + " to " + chunkName + " at " + newInfo.dataOffsets[0] + " to " + newInfo.dataOffsets[1]
                );

                Map<String, TensorInfo> tensors = tensorsInChunk.computeIfAbsent(chunkName, n -> new LinkedHashMap<>());
                tensors.put(name, newInfo);
            }

            // Now We have the data im place data, write the real file
            for (Map.Entry<String, Pair<RandomAccessFile, FileChannel>> entry : chunkFiles.entrySet()) {
                String chunkName = entry.getKey();
                Pair<RandomAccessFile, FileChannel> chunkFile = entry.getValue();

                FileChannel ch = chunkFile.left.getChannel();
                Map<String, TensorInfo> chunkTensors = tensorsInChunk.get(chunkName);

                byte[] header = om.writeValueAsBytes(chunkTensors);
                System.out.println("Writing " + chunkName + " with " + chunkTensors.size() + " tensors");
                // System.out.println(new String(header));
                byte[] hsize = new byte[Long.BYTES];
                ByteBuffer.wrap(hsize).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length);

                try (RandomAccessFile raf = new RandomAccessFile(Paths.get(modelDir, chunkName).toFile(), "rw")) {
                    raf.write(hsize);
                    raf.write(header);
                    raf.seek(raf.length());
                    System.out.println("Writing " + ch.size() + " bytes of data from " + raf.getChannel().position());
                    ch.transferTo(0, ch.size(), raf.getChannel());
                }
            }

            // Write the index
            try (RandomAccessFile raf = new RandomAccessFile(Paths.get(modelDir, SafeTensorIndex.MODEL_INDEX_JSON).toFile(), "rw")) {
                raf.write(om.writeValueAsBytes(Map.of("metadata", new HashMap<>(), "weight_map", tensorIndex)));
            }

            // Clean up
            for (Pair<RandomAccessFile, FileChannel> p : chunkFiles.values()) {
                p.left.close();
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
