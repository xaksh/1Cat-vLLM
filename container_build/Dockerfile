FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Ubuntu 24.04 ships glibc 2.39, satisfying 1Cat-vLLM v1.5.0's GLIBC_2.38
# requirement for its flash_attn_v100_cuda extension (Ubuntu 20.04 host has 2.31).
# devel (not runtime) base needed for a real nvcc — the pip nvidia-cuda-nvcc-cu12
# package only ships ptxas/nvvm, not the nvcc compiler driver itself.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch==2.10.* --index-url https://download.pytorch.org/whl/cu128

COPY 1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl \
    && rm /tmp/1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl

WORKDIR /workspace
