#!/bin/bash
# Install flash-attn in spacetools conda environments.
# Must be run on a GPU node with at least 1 GPU.
#
# Usage:
#   (allocate a GPU node, e.g. via salloc or sbatch)
#   bash install_flash_attn.sh [sft|rl|both]

set -e

TARGET="${1:-both}"

install_flash_attn() {
    local ENV_NAME="$1"
    echo "=== Installing flash-attn in $ENV_NAME ==="
    
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    # Install CUDA toolkit if nvcc not available
    if ! which nvcc &>/dev/null; then
        echo "Installing CUDA toolkit via conda..."
        conda install -c nvidia cuda-toolkit=12.8 -y 2>&1 | tail -3
    fi
    
    export CUDA_HOME="$CONDA_PREFIX"
    echo "CUDA_HOME=$CUDA_HOME"
    
    # Build and install flash-attn (takes 20-45 minutes)
    echo "Building flash-attn (this takes 20-45 minutes)..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -5
    
    # Verify
    python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__} installed successfully')" 2>/dev/null || echo "WARNING: flash-attn import failed"
    
    conda deactivate
}

case "$TARGET" in
    sft)
        install_flash_attn spacetools-sft
        ;;
    rl)
        install_flash_attn spacetools-rl
        ;;
    both)
        install_flash_attn spacetools-sft
        install_flash_attn spacetools-rl
        ;;
    *)
        echo "Usage: bash install_flash_attn.sh [sft|rl|both]"
        exit 1
        ;;
esac

echo "=== Done ==="
