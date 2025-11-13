# Download TVM pre-built binaries from GitHub Release

# Configuration
$TVM_VERSION = if ($env:TVM_VERSION) { $env:TVM_VERSION } else { "0.22.0" }
$RELEASE_TAG = "tvm-v$TVM_VERSION"

# Get project root directory (git root)
try {
    $PROJECT_ROOT = git rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -ne 0) {
        $PROJECT_ROOT = Get-Location
    }
} catch {
    $PROJECT_ROOT = Get-Location
}

$DOWNLOAD_DIR = Join-Path $PROJECT_ROOT "third_party\tvm"

# Auto-detect repository from git remote
try {
    $REMOTE_URL = git config --get remote.origin.url 2>$null
    if ($REMOTE_URL -match 'github\.com[:/]([^/]+)/([^/]+?)(\.git)?$') {
        $REPO_OWNER = $matches[1]
        $REPO_NAME = $matches[2]
    } else {
        Write-Error "Error: Could not detect GitHub repository"
        exit 1
    }
} catch {
    Write-Error "Error: Could not detect GitHub repository"
    exit 1
}

Write-Host "Downloading TVM v$TVM_VERSION from $REPO_OWNER/$REPO_NAME"

# Create download directory
New-Item -ItemType Directory -Force -Path $DOWNLOAD_DIR | Out-Null

# Detect platform (PowerShell is typically Windows, but check to be sure)
$PLATFORM = "Windows"
$BUILD_FILE = "tvm-windows-x64.zip"

Write-Host "Detected platform: $PLATFORM"

# GitHub Release URL
$BASE_URL = "https://github.com/$REPO_OWNER/$REPO_NAME/releases/download/$RELEASE_TAG"

# Download build for current platform
$BUILD_PATH = Join-Path $DOWNLOAD_DIR $BUILD_FILE
Write-Host "Downloading $BUILD_FILE..."
try {
    Invoke-WebRequest -Uri "$BASE_URL/$BUILD_FILE" -OutFile $BUILD_PATH -ErrorAction Stop
    Write-Host "✓ $PLATFORM build downloaded" -ForegroundColor Green
} catch {
    Write-Host "✗ $PLATFORM build not found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Downloads saved to: $DOWNLOAD_DIR"

# List downloaded files
$files = Get-ChildItem -Path $DOWNLOAD_DIR -Filter "tvm-*" -ErrorAction SilentlyContinue
if ($files) {
    $files | Format-Table Name, Length, LastWriteTime -AutoSize
} else {
    Write-Host "No files downloaded"
}
