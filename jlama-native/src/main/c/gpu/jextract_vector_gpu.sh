#!/bin/bash


# Generate Java source code
/usr/local/jextract-20/bin/jextract --source \
  --output ../../java20 \
  -t com.github.tjake.jlama.tensor.operations.gpunative \
  -I . \
  -l jlamagpu \
  --header-class-name NativeGPU \
  vector_gpu.h

/usr/local/jextract-21/bin/jextract --source \
  --output ../../java21 \
  -t com.github.tjake.jlama.tensor.operations.gpunative \
  -I . \
  -l jlamagpu \
  --header-class-name NativeGPU \
  vector_gpu.h

/usr/local/jextract-22/bin/jextract \
  --output ../../java22 \
  -t com.github.tjake.jlama.tensor.operations.gpunative \
  -I . \
  -l jlamagpu \
  --header-class-name NativeGPU \
  vector_gpu.h
