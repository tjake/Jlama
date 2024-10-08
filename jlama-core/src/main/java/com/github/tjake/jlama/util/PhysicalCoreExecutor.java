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

import com.google.common.base.Suppliers;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

/**
 * Executor that uses a fixed number of physical cores
 */
public class PhysicalCoreExecutor {
    private static volatile int physicalCoreCount = Math.max(2, Runtime.getRuntime().availableProcessors() / 2);
    private static final AtomicBoolean started = new AtomicBoolean(false);

    /**
     * Override the number of physical cores to use
     * @param threadCount number of physical cores to use
     */
    public static void overrideThreadCount(int threadCount) {
        assert threadCount > 0 && threadCount <= Runtime.getRuntime().availableProcessors() : "Threads must be < cores: " + threadCount;

        if (!started.compareAndSet(false, true)) throw new IllegalStateException("Executor already started");

        physicalCoreCount = threadCount;
    }

    public static final Supplier<PhysicalCoreExecutor> instance = Suppliers.memoize(() -> {
        started.set(true);
        return new PhysicalCoreExecutor(physicalCoreCount);
    });

    private final ForkJoinPool pool;

    private PhysicalCoreExecutor(int cores) {
        assert cores > 0 && cores <= Runtime.getRuntime().availableProcessors() : "Invalid core count: " + cores;
        this.pool = new ForkJoinPool(cores, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true);
    }

    public void execute(Runnable run) {
        pool.submit(run).join();
    }

    public <T> T submit(Supplier<T> run) {
        return pool.submit(run::get).join();
    }

    public int getCoreCount() {
        return pool.getParallelism();
    }
}
