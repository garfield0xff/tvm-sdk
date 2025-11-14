#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' 

# Project Root Directory
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TVM SDK Docker Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Build Docker IMG
echo -e "${YELLOW}Building Docker image...${NC}"
docker-compose build

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Docker image built successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To enter the container:"
echo -e "  ${BLUE}docker-compose run --rm tvm-sdk bash${NC}"
echo ""
echo "Inside the container, you can:"
echo -e "  ${BLUE}# Build the project${NC}"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON"
echo "  cmake --build . -j\$(nproc)"
echo ""
echo -e "  ${BLUE}# Run examples${NC}"
echo "  ./examples/simple_tvm_import"
echo "  ./examples/tvm_version_example"
echo ""
echo -e "  ${BLUE}# Or install and run${NC}"
echo "  cmake --install . --prefix /usr/local"
echo "  /usr/local/bin/simple_tvm_import"
