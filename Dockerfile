FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies and LLVM 21
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    gnupg \
    lsb-release \
    ninja-build \
    python3-dev \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-21 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y llvm-21 llvm-21-dev \
    && rm -rf /var/lib/apt/lists/*

# Set LLVM environment variables
ENV LLVM_CONFIG=/usr/bin/llvm-config-21
ENV LD_LIBRARY_PATH=/usr/lib/llvm-21/lib:$LD_LIBRARY_PATH

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    torch \
    torchvision

# Copy project files
COPY . /workspace

# Download and extract Linux TVM wheel if exists
RUN if [ -f "whl/tvm/tvm-linux-x64.tar.gz" ]; then \
        mkdir -p whl/tvm/linux && \
        tar -xzf whl/tvm/tvm-linux-x64.tar.gz -C whl/tvm/linux && \
        pip install whl/tvm/linux/*.whl; \
    fi

# Add TVM libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/tvm

# Set Python path to ensure TVM is found
ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages:$PYTHONPATH

# Set TVM SDK Python path for custom modules
ENV TVM_SDK_PYTHON_PATH=/workspace/python

CMD ["/bin/bash"]