#!/bin/bash

gcc -fPIC -O3 -march=native -shared -o libjlamav.so vector_simd.c

# Generate Java source code
jextract --source \
  --output ../java \
  -t com.github.tjake.jlama.math.panama \
  -I . \
  -l jlamav \
  --header-class-name VectorNativeSimd \
  vector_simd.h