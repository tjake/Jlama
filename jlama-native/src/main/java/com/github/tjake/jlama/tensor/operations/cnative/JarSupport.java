package com.github.tjake.jlama.tensor.operations.cnative;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.tjake.jlama.util.RuntimeSupport;

public class JarSupport {
    private static final Logger logger = LoggerFactory.getLogger(JarSupport.class);
    static boolean maybeLoadLibrary() {
        String ext = RuntimeSupport.isMac() ? ".dylib" : RuntimeSupport.isWin() ? ".dll" : ".so";
        URL lib = JarSupport.class.getClassLoader().getResource("META-INF/native/lib/libjlama" + ext);

        if (lib != null) {
            try {
                final File libpath = Files.createTempDirectory("jlama").toFile();
                libpath.deleteOnExit(); // just in case

                File libfile = Paths.get(libpath.getAbsolutePath(), "libjlama"+ext).toFile();
                libfile.deleteOnExit(); // just in case

                final InputStream in = lib.openStream();
                final OutputStream out = new BufferedOutputStream(new FileOutputStream(libfile));

                int len = 0;
                byte[] buffer = new byte[8192];
                while ((len = in.read(buffer)) > -1)
                    out.write(buffer, 0, len);
                out.close();
                in.close();
                System.load(libfile.getAbsolutePath());
                logger.debug("Loaded jlama-native library: {}", libfile.getAbsolutePath());
                return true;
            } catch (IOException e) {
                logger.warn("Error loading jlama-native library");
            }
        }

        logger.warn("jlama-native shared library not found");
        return false;
    }

}
