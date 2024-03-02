package com.github.tjake.jlama.util;

/**
 * Represents an operation that accepts three input arguments and returns no
 * result.  This is the three-arity specialization of {@link java.util.function.Consumer}.
 * Unlike most other functional interfaces, {@code TriConsumer} is expected
 * to operate via side-effects.
 *
 * <p>This is a <a href="package-summary.html">functional interface</a>
 * whose functional method is {@link #accept(Object, Object, Object)}.
 *
 * @param <P> the type of the first argument to the operation
 * @param <M> the type of the second argument to the operation
 * @param <N> the type of the third argument to the operation
 */
@FunctionalInterface
public interface TriConsumer<P,M,N> {
    void accept(P p1, M p2, N p3);
}
