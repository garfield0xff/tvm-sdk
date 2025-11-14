# Setup Docker Image for TVM SDK
# Usage: .\script\setup-docker.ps1

$ErrorActionPreference = "Stop"

# Get project root directory
try {
    $PROJECT_ROOT = git rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -ne 0) {
        $PROJECT_ROOT = Get-Location
    }
} catch {
    $PROJECT_ROOT = Get-Location
}

Set-Location $PROJECT_ROOT

Write-Host "========================================" -ForegroundColor Blue
Write-Host "TVM SDK Docker Setup" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Build Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker-compose build

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Docker image built successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
