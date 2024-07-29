package com.github.tjake.jlama.util;

import com.github.tjake.jlama.tensor.AbstractTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DebugSupport {
    private static final boolean DEBUG = false;
    private static final Logger logger = LoggerFactory.getLogger(DebugSupport.class);

    public static boolean isDebug() {
        return DEBUG;
    }

    public static void debug(String name, AbstractTensor t, int layer) {
        if (DEBUG) {
            logger.debug("Layer: {} - {} - {}", layer, name, t);
        }
    }

    public static void debug(String msg, Object t) {
        if (DEBUG) {
            logger.debug("{} - {}", msg, t);
        }
    }
}
