FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies including LLVM
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    llvm \
    llvm-dev \
    && rm -rf /var/lib/apt/lists/*

# Set LLVM environment variables
ENV LLVM_CONFIG=/usr/bin/llvm-config

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    torch \
    torchvision

# Copy project files
COPY . /workspace

# Download and extract Linux TVM wheel if exists
RUN if [ -f "third_party/tvm/tvm-linux-x64.tar.gz" ]; then \
        mkdir -p third_party/tvm/linux && \
        tar -xzf third_party/tvm/tvm-linux-x64.tar.gz -C third_party/tvm/linux && \
        pip install third_party/tvm/linux/*.whl; \
    fi

CMD ["/bin/bash"]
