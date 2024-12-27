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
package com.github.tjake.jlama.tensor.operations.util;

import com.github.tjake.jlama.util.RuntimeSupport;
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

public class JarSupport {
    private static final Logger logger = LoggerFactory.getLogger(JarSupport.class);

    public static boolean maybeLoadLibrary(String libname) {
        String ext = RuntimeSupport.isMac() ? ".dylib" : RuntimeSupport.isWin() ? ".dll" : ".so";
        URL lib = JarSupport.class.getClassLoader().getResource("META-INF/native/lib/lib" + libname + ext);

        if (lib != null) {
            try {
                final File libpath = Files.createTempDirectory("jlama").toFile();
                libpath.deleteOnExit(); // just in case

                File libfile = Paths.get(libpath.getAbsolutePath(), "lib" + libname + ext).toFile();
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
                logger.debug("Loaded {}-native library: {}", libname, libfile.getAbsolutePath());
                return true;
            } catch (IOException e) {
                logger.warn("Error loading {}-native library", libname);
            }
        }

        logger.warn("jlama-native shared library not found");
        return false;
    }
}
