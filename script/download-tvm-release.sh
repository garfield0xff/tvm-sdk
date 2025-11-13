#!/bin/bash

# Download TVM pre-built binaries from GitHub Release

set -e

# Configuration
TVM_VERSION="${TVM_VERSION:-0.22.0}"
RELEASE_TAG="tvm-v${TVM_VERSION}"
DOWNLOAD_DIR="./dist"

# Auto-detect repository from git remote
REMOTE_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+)/([^/]+)(\.git)?$ ]]; then
    REPO_OWNER="${BASH_REMATCH[1]}"
    REPO_NAME="${BASH_REMATCH[2]%.git}"
else
    echo "Error: Could not detect GitHub repository"
    exit 1
fi

echo "Downloading TVM v${TVM_VERSION} from ${REPO_OWNER}/${REPO_NAME}"

# Create download directory
mkdir -p "${DOWNLOAD_DIR}"

# GitHub Release URL
BASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}"

# Download Windows build
WINDOWS_FILE="tvm-windows-x64.zip"
echo "Downloading ${WINDOWS_FILE}..."
if curl -L -f -o "${DOWNLOAD_DIR}/${WINDOWS_FILE}" "${BASE_URL}/${WINDOWS_FILE}"; then
    echo "✓ Windows build downloaded"
else
    echo "⚠ Windows build not found"
fi

# Download macOS build
MACOS_FILE="tvm-macos-universal.tar.gz"
echo "Downloading ${MACOS_FILE}..."
if curl -L -f -o "${DOWNLOAD_DIR}/${MACOS_FILE}" "${BASE_URL}/${MACOS_FILE}"; then
    echo "✓ macOS build downloaded"
else
    echo "⚠ macOS build not found"
fi

echo ""
echo "Downloads saved to: ${DOWNLOAD_DIR}"
ls -lh "${DOWNLOAD_DIR}"/tvm-* 2>/dev/null || echo "No files downloaded"
