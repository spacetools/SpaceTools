#!/bin/bash
# SpaceTools Environment Setup
# Usage: source setup_envs.sh [sft|rl|toolshed|all]
#
# Requires: conda, CUDA-capable GPU node (for flash-attn compilation)
# Run on a GPU node (8 GPUs recommended)
#
# Version pins (shared across SFT and RL for checkpoint compatibility):
#   Python:       3.11
#   transformers: 4.57.1
#   torch:        2.9.1  (from sglang 0.5.6)
#   ray:          2.47.1 (shared with Toolshed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PREFIX_DIR="$(conda info --base)/envs"

# Shared version pins
PYTHON_VERSION=3.11
TRANSFORMERS_VERSION=4.57.1
TORCH_VERSION=2.9.1
RAY_VERSION=2.47.1

echo "=== SpaceTools Environment Setup ==="
echo "Python:       $PYTHON_VERSION"
echo "transformers: $TRANSFORMERS_VERSION"
echo "torch:        $TORCH_VERSION"
echo "ray:          $RAY_VERSION"
echo ""

setup_sft() {
    echo "========================================="
    echo "Setting up spacetools-sft environment"
    echo "========================================="

    conda create -n spacetools-sft python==$PYTHON_VERSION -y
    conda activate spacetools-sft

    # Install torch first (controls CUDA version)
    pip install torch==$TORCH_VERSION torchvision torchaudio

    # Install transformers at pinned version
    pip install transformers==$TRANSFORMERS_VERSION

    # Install SpaceTools-SFT in editable mode (will respect already-installed torch/transformers)
    cd "$SCRIPT_DIR/SpaceTools-SFT"
    pip install -e ".[torch,metrics]" --no-deps 2>/dev/null || pip install -e "." --no-deps
    # Install remaining SpaceTools-SFT deps (excluding torch/transformers which are already installed)
    pip install datasets accelerate "peft>=0.18.0" "trl>=0.18.0" torchdata \
        gradio matplotlib "tyro<0.9.0" \
        einops numpy pandas scipy \
        sentencepiece tiktoken modelscope hf-transfer safetensors \
        av fire omegaconf packaging protobuf pyyaml pydantic \
        uvicorn fastapi sse-starlette \
        deepspeed

    # Install flash-attn (needs GPU node for compilation)
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: flash-attn install failed. Make sure you're on a GPU node."

    echo ""
    echo "=== spacetools-sft setup complete ==="
    echo "Verify: conda activate spacetools-sft && python -c 'import llamafactory; import torch; print(torch.__version__)'"

    conda deactivate
}

setup_rl() {
    echo "========================================="
    echo "Setting up spacetools-rl environment"
    echo "========================================="

    conda create -n spacetools-rl python==$PYTHON_VERSION -y
    conda activate spacetools-rl

    # Install torch first
    pip install torch==$TORCH_VERSION torchvision torchaudio

    # Install transformers at pinned version
    pip install transformers==$TRANSFORMERS_VERSION

    # Install sglang (will use already-installed torch)
    pip install "sglang[srt,openai]==0.5.6" --no-deps
    # Install sglang deps manually (excluding torch/transformers already installed)
    pip install sglang_router flashinfer_python compressed_tensors \
        outlines lm-format-enforcer cuda-python \
        aiohttp anthropic openai blobfile decord2 \
        torchao pillow fastapi uvicorn pydantic requests \
        numpy scipy packaging psutil filelock triton

    # Install ray (shared version with Toolshed)
    pip install "ray[default]==$RAY_VERSION"

    # Install SpaceTools-RL in editable mode
    cd "$SCRIPT_DIR/SpaceTools-RL"
    pip install -e "." --no-deps
    # Install verl deps (excluding torch/transformers/ray already installed)
    pip install accelerate codetiming datasets dill hydra-core \
        "numpy<2.0.0" pandas peft "pyarrow>=19.0.0" pybind11 pylatexenc \
        torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
        wandb tensorboard packaging

    # Install Toolshed (for verl integration - toolshed.integration.verl)
    cd "$SCRIPT_DIR/SpaceTools-Toolshed"
    pip install -e . --no-deps
    # Install toolshed core deps (excluding ray already installed)
    pip install docstring_parser aiohttp aiohttp-cors Pillow \
        fastapi uvicorn python-multipart openai botocore pyyaml \
        anthropic matplotlib scipy requests click uvloop

    # Install flash-attn
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: flash-attn install failed. Make sure you're on a GPU node."

    # Install qwen_vl_utils for VLM support
    pip install qwen_vl_utils

    echo ""
    echo "=== spacetools-rl setup complete ==="
    echo "Verify: conda activate spacetools-rl && python -c 'import verl; import toolshed; import torch; print(torch.__version__)'"

    conda deactivate
}

# Parse args
#
# Note: There is no separate spacetools-toolshed env. The spacetools-rl env includes
# toolshed + ray and serves as the base for creating per-tool envs via
# SpaceTools-Toolshed/install_tools/setup_tool_env.sh (which detects Python + Ray
# from the active env).
case "${1:-all}" in
    sft)
        setup_sft
        ;;
    rl)
        setup_rl
        ;;
    all)
        setup_sft
        setup_rl
        ;;
    *)
        echo "Usage: source setup_envs.sh [sft|rl|all]"
        ;;
esac
