package com.github.tjake.jlama.util;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

import com.google.common.base.Suppliers;

public class PhysicalCoreExecutor {
    private static volatile int physicalCoreCount = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final AtomicBoolean started = new AtomicBoolean(false);
    public static void overrideThreadCount(int threadCount) {
        if (!started.compareAndSet(false, true))
            throw new IllegalStateException("Executor already started");

        physicalCoreCount = threadCount;
    }

    public static final Supplier<PhysicalCoreExecutor> instance = Suppliers.memoize(() -> {
        started.set(true);
        return new PhysicalCoreExecutor(physicalCoreCount);
    });

    private final ForkJoinPool pool;

    private PhysicalCoreExecutor(int cores) {
        assert cores > 0 && cores <= Runtime.getRuntime().availableProcessors() : "Invalid core count: " + cores;
        this.pool = new ForkJoinPool(cores);
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
