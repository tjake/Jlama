#
# The Jlama Project licenses this file to you under the Apache License,
# version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at:
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

## GNU Makefile designed to build a static library which can be shared across multiple architectures.

## Input environment:
# CC - compiler (gcc or clang)
# AR - archiver (ar)
# JNI_PLATFORM - "linux" for linux and "darwin" for mac.
# LIB_DIR - where the static library will be built in
# OBJ_DIR - where the obj files will be built in (defaults to LIB_DIR)
# LIB_NAME - the name of the native library
# LIB_EXT - the extension of the native library

SRC_DIR = src/main/c
LIB = $(LIB_DIR)/$(LIB_NAME).$(LIB_EXT)

SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(LIB)

$(LIB): $(OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) $(LDFLAGS) -o $(LIB) $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR)
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(LIB_DIR) $(OBJ_DIR)

## Debug support
# use make print-VARIABLE name to see the value
print-%  : ; @echo $* = $($*)
