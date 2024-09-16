///usr/bin/env jbang "$0" "$@" ; exit $?
//COMPILE_OPTIONS -source 20
//RUNTIME_OPTIONS -server -Dstdout.encoding=UTF-8 -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 --add-modules=jdk.incubator.vector --add-exports java.base/sun.nio.ch=ALL-UNNAMED --enable-preview --enable-native-access=ALL-UNNAMED -XX:+UnlockDiagnosticVMOptions -XX:CompilerDirectivesFile=./inlinerules.json -XX:+AlignVector -XX:+UseStringDeduplication -XX:+UseCompressedOops -XX:+UseCompressedClassPointers

//DEPS com.github.tjake:jlama-cli:0.4.0
//DEPS com.github.tjake:jlama-native:0.4.0:${os.detected.name}-${os.detected.arch}

import static java.lang.System.*;
import com.github.tjake.jlama.cli.JlamaCli;

/**
 * JBANG! script for running jlama-cli
 */
public class jlama {
    public static void main(String... args) {
        JlamaCli.main(args);
    }
}
