#!/bin/bash

gcc -fPIC -O3 -march=native -shared -o libjlama.so vector_simd.c

# Generate Java source code
jextract --source \
  --output ../java \
  -t com.github.tjake.jlama.tensor.operations.cnative \
  -I . \
  -l jlama \
  --header-class-name NativeSimd \
  vector_simd.h