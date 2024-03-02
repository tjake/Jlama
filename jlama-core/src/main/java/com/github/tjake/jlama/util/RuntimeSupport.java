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

/**
 * Helper class for runtime support
 */
public class RuntimeSupport {
    static String OS = System.getProperty("os.name").toLowerCase();
    static String Arch = System.getProperty("os.arch").toLowerCase();
    static String ArchBits = System.getProperty("sun.arch.data.model");

    public static boolean isLinux() {
        return OS.contains("linux");
    }

    public static boolean isMac() {
        return OS.contains("mac");
    }

    public static boolean isWin() {
        return OS.contains("win");
    }

    public static boolean isX86() {
        return Arch.contains("x86");
    }

    public static boolean isArm() {
        return Arch.contains("aarch");
    }

    public static boolean is32Bit() {
        return ArchBits.contains("32");
    }

    public static boolean is64Bit() {
        return ArchBits.contains("64");
    }
}
