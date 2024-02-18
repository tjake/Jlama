package com.github.tjake.jlama.util;

@FunctionalInterface
public interface TriConsumer<P,M,N> {
    void accept(P p1, M p2, N p3);
}
