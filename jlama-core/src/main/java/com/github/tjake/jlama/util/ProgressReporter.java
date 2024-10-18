package com.github.tjake.jlama.util;

public interface ProgressReporter {

    void update(String filename, long sizeDownloaded, long totalSize);

}
