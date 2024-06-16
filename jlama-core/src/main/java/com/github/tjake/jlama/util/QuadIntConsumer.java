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
 * Represents an operation that accepts four input arguments and returns no
 * result.  This is the four-arity specialization of {@link java.util.function.Consumer}.
 * Unlike most other functional interfaces, {@code QuadConsumer} is expected
 * to operate via side-effects.
 *
 * <p>This is a <a href="package-summary.html">functional interface</a>
 * whose functional method is {@link #accept(int, int, int, int)}.
 *
 * @param p1 the type of the first argument to the operation
 * @param p2 the type of the second argument to the operation
 * @param p3 the type of the third argument to the operation
 * @param p4 the type of the fourth argument to the operation
 */
@FunctionalInterface
public interface QuadIntConsumer {
    void accept(int p1, int p2, int p3, int p4);
}
