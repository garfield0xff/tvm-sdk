FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.9
RUN conda create -n tvm-sdk python=3.9 -y

# Activate environment and install conda packages
SHELL ["conda", "run", "-n", "tvm-sdk", "/bin/bash", "-c"]

# Install LLVM and other dependencies via conda
RUN conda install -c conda-forge llvmdev -y

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    torch \
    torchvision

# Copy project files
COPY . /workspace

# Download and extract Linux TVM wheel
RUN if [ -f "script/download-tvm-release.sh" ]; then \
        chmod +x script/download-tvm-release.sh && \
        ./script/download-tvm-release.sh || true; \
    fi && \
    if [ -f "third_party/tvm/tvm-linux-x64.tar.gz" ]; then \
        mkdir -p third_party/tvm/linux && \
        tar -xzf third_party/tvm/tvm-linux-x64.tar.gz -C third_party/tvm/linux && \
        pip install third_party/tvm/linux/*.whl; \
    fi

# Set default command to activate conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tvm-sdk"]
CMD ["/bin/bash"]
