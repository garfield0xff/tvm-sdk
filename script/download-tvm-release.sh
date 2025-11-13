#!/bin/bash

# Download TVM pre-built binaries from GitHub Release

set -e

# Configuration
TVM_VERSION="${TVM_VERSION:-0.22.0}"
RELEASE_TAG="tvm-v${TVM_VERSION}"

# Get project root directory (git root)
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
DOWNLOAD_DIR="${PROJECT_ROOT}/third_party/tvm"

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

# Detect OS
OS_TYPE=$(uname -s)
case "${OS_TYPE}" in
    Linux*)
        PLATFORM="Linux"
        BUILD_FILE="tvm-linux-x64.tar.gz"
        ;;
    Darwin*)
        PLATFORM="macOS"
        BUILD_FILE="tvm-macos-universal.tar.gz"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        PLATFORM="Windows"
        BUILD_FILE="tvm-windows-x64.zip"
        ;;
    *)
        echo "Error: Unsupported OS: ${OS_TYPE}"
        exit 1
        ;;
esac

echo "Detected platform: ${PLATFORM}"

# GitHub Release URL
BASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}"

# Download build for current platform
echo "Downloading ${BUILD_FILE}..."
if curl -L -f -o "${DOWNLOAD_DIR}/${BUILD_FILE}" "${BASE_URL}/${BUILD_FILE}"; then
    echo "✓ ${PLATFORM} build downloaded"
else
    echo "✗ ${PLATFORM} build not found"
    exit 1
fi

echo ""
echo "Downloads saved to: ${DOWNLOAD_DIR}"
ls -lh "${DOWNLOAD_DIR}"/tvm-* 2>/dev/null || echo "No files downloaded"
