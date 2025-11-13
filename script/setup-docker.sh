#!/bin/bash

# Setup and run TVM SDK Docker environment

set -e

echo "Setting up TVM SDK Docker environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo "Error: docker-compose is not available"
    echo "Please install docker-compose"
    exit 1
fi

# Use 'docker compose' or 'docker-compose' depending on availability
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
fi

echo "Building Docker image..."
$DOCKER_COMPOSE build

echo ""
echo "Docker environment ready!"
echo ""
echo "Usage:"
echo "  Start container:    $DOCKER_COMPOSE up -d"
echo "  Enter container:    $DOCKER_COMPOSE exec tvm-sdk /bin/bash"
echo "  Run command:        $DOCKER_COMPOSE run --rm tvm-sdk python your_script.py"
echo "  Stop container:     $DOCKER_COMPOSE down"
echo ""
echo "To download TVM wheels inside container:"
echo "  $DOCKER_COMPOSE run --rm tvm-sdk ./script/download-tvm-release.sh"
