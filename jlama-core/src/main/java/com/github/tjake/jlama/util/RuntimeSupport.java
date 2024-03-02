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
