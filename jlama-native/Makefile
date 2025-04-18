#
# The Jlama Project licenses this file to you under the Apache License,
# version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at:
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

## GNU Makefile designed to build a static library which can be shared across multiple architectures.

## Input environment:
#   CC           - compiler (gcc or clang)
#   AR           - archiver (ar)
#   JNI_PLATFORM - "linux" for linux and "darwin" for mac.
#   LIB_DIR      - where the static library will be built in
#   OBJ_DIR      - where the obj files will be built in (defaults to LIB_DIR)
#   LIB_NAME     - the name of the native library
#   LIB_EXT      - the extension of the native library (e.g. a / dll)
OBJ_DIR ?= $(LIB_DIR)

#───────────────────────────────────────────────────────────────────────────────
# Detect Windows targets
ifeq ($(OS),Windows_NT)
  WINDOWS := 1
endif

#───────────────────────────────────────────────────────────────────────────────
# SIMD‐only build (all platforms)
SRC_DIR := src/main/c/simd
SRCS    := $(wildcard $(SRC_DIR)/*.c)
OBJS    := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
LIB     := $(LIB_DIR)/$(LIB_NAME).$(LIB_EXT)

# GPU‐accelerated build (optional; enabled if CFLAGS2 is set)
OBJ2_DIR := $(OBJ_DIR)/gpu
SRC2_DIR := src/main/c/gpu
SRCS2    := $(wildcard $(SRC2_DIR)/*.c)
OBJS2    := $(SRCS2:$(SRC2_DIR)/%.c=$(OBJ2_DIR)/%.o)
LIB2     := $(LIB_DIR)/$(LIB_NAME)gpu.$(LIB_EXT)

all: $(LIB)
ifdef CFLAGS2
all: $(LIB2)
endif

#───────────────────────────────────────────────────────────────────────────────
# 1) Build the core SIMD library (static .a on all platforms)
$(LIB): $(OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) $(LDFLAGS) -o $(LIB) $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR)
	$(CC) -c $< -o $@ $(CFLAGS)

#───────────────────────────────────────────────────────────────────────────────
# 2) GPU target: either DLL on Windows or static .a on others
ifdef WINDOWS

  # Where your Dawn runtime DLL lives; override if needed:
  WEBGPU_DLL  ?= path/to/webgpu_dawn.dll
  WEBGPU_DEF  := $(OBJ2_DIR)/webgpu_dawn.def
  WEBGPU_A    := $(OBJ2_DIR)/libwebgpu_dawn.a

  # 2.a) Extract .def from the Dawn DLL
  $(WEBGPU_DEF): $(WEBGPU_DLL)
  	mkdir -p $(OBJ2_DIR)
  	gendef "$<"
  	mv webgpu_dawn.def "$@"

  # 2.b) Turn that .def into a MinGW import‐lib
  $(WEBGPU_A): $(WEBGPU_DEF)
  	dlltool --dllname webgpu_dawn.dll --def "$<" --output-lib "$@"

  # 2.c) Link your GPU code + Dawn import lib into a real Windows DLL
  $(LIB2): $(WEBGPU_A) $(OBJS2)
  	mkdir -p $(LIB_DIR)
  	$(CC) -shared \
  	  -o "$@" \
  	  $(OBJS2) $(WEBGPU_A) \
  	  -Wl,--out-implib,$(LIB_DIR)/lib$(LIB_NAME)gpu.a \
  	  -lmingw32 -lmingwex -lmsvcrt

else

  # Non‑Windows: just produce a static archive for GPU bits too
  $(LIB2): $(OBJS2)
  	mkdir -p $(LIB_DIR)
  	$(AR) $(LDFLAGS2) -o $(LIB2) $^

endif

# Compile GPU object files
$(OBJ2_DIR)/%.o: $(SRC2_DIR)/%.c
	mkdir -p $(OBJ2_DIR)
	$(CC) -c $< -o $@ $(CFLAGS2)

#───────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(LIB_DIR) $(OBJ_DIR)

## Debug support
# use `make print-VARIABLE name` to see the value of any variable
print-%  : ; @echo $* = $($*)

