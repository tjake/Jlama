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
