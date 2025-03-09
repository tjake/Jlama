#!/usr/bin/env bash
#
# download_wgpu_native.sh
#
# This script downloads a specific wgpu-native ZIP asset from:
#   https://github.com/gfx-rs/wgpu-native/releases
#
# Usage:
#   ./download_wgpu_native.sh <TAG> <PLATFORM> <ARCH> <BUILD_TYPE> [TOOLCHAIN] [TARGET_DIR]
#
# Parameters:
#   TAG          - The release tag, e.g. "v22.1.0.5".
#   PLATFORM     - One of: "linux", "macos", "windows".
#   ARCH         - One of: "aarch64", "x86_64", "i686" (where applicable).
#   BUILD_TYPE   - "debug" or "release".
#   TOOLCHAIN    - (Required only for Windows) "msvc" or "gnu".
#   TARGET_DIR   - Directory to extract to (default: "./wgpu-native")
#
# Examples:
#   # Linux x86_64 debug build, extracts to ./wgpu-native
#   ./download_wgpu_native.sh v22.1.0.5 linux x86_64 debug
#
#   # MacOS aarch64 release build, extracts to ./my-lib
#   ./download_wgpu_native.sh v22.1.0.5 macos aarch64 release "" ./my-lib
#     (Note the empty string "" for TOOLCHAIN, since it's not needed on macOS)
#
#   # Windows x86_64 msvc release, extracts to ./wgpu-win
#   ./download_wgpu_native.sh v22.1.0.5 windows x86_64 release msvc wgpu-win
#
# Requirements:
#   - curl
#   - jq
#   - unzip

set -euo pipefail

# You can adjust these if you fork or use a different repository.
OWNER="gfx-rs"
REPO="wgpu-native"

# Read command line arguments
TAG="${1:-}"
PLATFORM="${2:-}"
ARCH="${3:-}"
BUILD_TYPE="${4:-}"
TOOLCHAIN="${5:-}"       # Only needed if PLATFORM=windows
TARGET_DIR="${6:-./wgpu-native}"

# Check mandatory parameters
if [ -z "$TAG" ] || [ -z "$PLATFORM" ] || [ -z "$ARCH" ] || [ -z "$BUILD_TYPE" ]; then
  echo "Usage: $0 <TAG> <PLATFORM> <ARCH> <BUILD_TYPE> [TOOLCHAIN] [TARGET_DIR]"
  echo "  TAG: e.g. v22.1.0.5"
  echo "  PLATFORM: linux, macos, windows"
  echo "  ARCH: x86_64, aarch64, i686 (depends on platform)"
  echo "  BUILD_TYPE: debug or release"
  echo "  TOOLCHAIN: (only required for Windows) msvc or gnu"
  echo "  TARGET_DIR: (optional) defaults to ./wgpu-native"
  exit 1
fi

# Construct the expected asset filename
# Format:
#   wgpu-<platform>-<arch>[-<toolchain>]-<build_type>.zip
#
# For example:
#   wgpu-linux-aarch64-debug.zip
#   wgpu-windows-x86_64-msvc-release.zip
#   wgpu-macos-x86_64-release.zip
#
# Windows builds require specifying the toolchain (gnu or msvc).
if [[ "$PLATFORM" == "windows" ]]; then
  if [ -z "$TOOLCHAIN" ]; then
    echo "Error: TOOLCHAIN is required for Windows (msvc or gnu)."
    exit 1
  fi
  ASSET_NAME="wgpu-${PLATFORM}-${ARCH}-${TOOLCHAIN}-${BUILD_TYPE}.zip"
else
  ASSET_NAME="wgpu-${PLATFORM}-${ARCH}-${BUILD_TYPE}.zip"
fi

echo "Downloading release asset '$ASSET_NAME' from tag '$TAG'..."

# Fetch release JSON via GitHub's API
API_URL="https://api.github.com/repos/${OWNER}/${REPO}/releases/tags/${TAG}"
RELEASE_JSON=$(curl -sS "$API_URL")

# Extract the asset's download URL using jq
ASSET_URL=$(echo "$RELEASE_JSON" | jq -r --arg name "$ASSET_NAME" '
  .assets[]
  | select(.name == $name)
  | .browser_download_url
')

# Check if the asset was found
if [ -z "$ASSET_URL" ] || [ "$ASSET_URL" == "null" ]; then
  echo "Error: No matching asset found for '$ASSET_NAME' in tag '$TAG'."
  echo "Check the spelling or ensure such an artifact exists in the release."
  exit 1
fi

TEMP_ZIP="/tmp/${ASSET_NAME}"
echo "Asset should be located (after download) at: $TEMP_ZIP"

# Check if we already have the ZIP in /tmp
if [ -f "$TEMP_ZIP" ]; then
  echo "File '$TEMP_ZIP' already exists. Skipping download..."
else
  echo "File '$TEMP_ZIP' not found. Downloading..."
  curl -L -o "$TEMP_ZIP" "$ASSET_URL"
fi

# Create the target directory if it does not exist
mkdir -p "$TARGET_DIR"

echo "Extracting '$ASSET_NAME' into '$TARGET_DIR'..."
unzip -o "$TEMP_ZIP" -d "$TARGET_DIR"

echo "Download and extraction complete!"
echo "Files have been placed in: $TARGET_DIR"

cp ~/workspace/jlama/webgpu.h $TARGET_DIR/include/webgpu/webgpu.h
cp ~/workspace/jlama/libwebgpu_dawn.so $TARGET_DIR/lib/libwebgpu_dawn.so
