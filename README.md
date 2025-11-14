# TVM SDK

C++ SDK for TVM (Apache TVM) 

## Prerequisites

- **Docker**

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd tvm-sdk
```

### 2. Build Docker image

**Linux/macOS:**
```bash
./script/setup-docker.sh
```

**Windows (PowerShell):**
```powershell
.\script\setup-docker.ps1
```

### 3. Test TVM installation

```bash
# Start container
docker-compose up -d

# Run TVM test
docker-compose exec tvm-sdk python3 test/tvm/tvm-test.py
```

