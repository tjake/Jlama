#!/usr/bin/env bash
#
# This script downloads a specific dawn-build ZIP asset from:
#   https://github.com/tjake/dawn-build/releases
#
# Usage:
#   ./download_dawn.sh <BUILD_TAG> <PLATFORM> <ARCH> [TARGET_DIR]
#
# Parameters:
#   TAG          - The release tag, e.g. "2025-03-23".
#   PLATFORM     - One of: "linux", "mac", "windows".
#   ARCH         - One of: "arm64", "x86_64", "x86" (where applicable).
#   TARGET_DIR   - Directory to extract to (default: "./dawn-native")
#
# Examples:
#   # Linux x86_64 debug build, extracts to ./dawn-native
#   ./download_dawn.sh 2025-03-23 linux x86_64
#
# Requirements:
#   - curl
#   - jq
#   - unzip

set -euo pipefail

# You can adjust these if you fork or use a different repository.
OWNER="tjake"
REPO="build-dawn"

# Read command line arguments
TAG="${1:-}"
PLATFORM="${2:-}"
ARCH="${3:-}"
TARGET_DIR="${4:-./dawn-native}"

# Check mandatory parameters
if [ -z "$TAG" ] || [ -z "$PLATFORM" ] || [ -z "$ARCH" ]; then
  echo "Usage: $0 <TAG> <PLATFORM> <ARCH>  [TARGET_DIR]"
  echo "  BUILD_TAG: e.g. 2025-03-23"
  echo "  PLATFORM: linux, mac, windows"
  echo "  ARCH: x86_64, arm64, x86 (depends on platform)"
  echo "  TARGET_DIR: (optional) defaults to ./dawn-native"
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
ASSET_NAME="dawn-${PLATFORM}-${ARCH}-${TAG}.zip"
echo "Downloading release asset '$ASSET_NAME'..."

# Fetch release JSON via GitHub's API
API_URL="https://api.github.com/repos/${OWNER}/${REPO}/releases/tags/${TAG}"
RELEASE_JSON=$(curl -sS "$API_URL")
echo "$RELEASE_JSON"

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

