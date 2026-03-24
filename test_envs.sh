#!/bin/bash
# Quick validation of SpaceTools environments
# Run on a GPU node

set -e

echo "=== Testing spacetools-sft ==="
conda run -n spacetools-sft python -c "
import torch
import transformers
print(f'torch={torch.__version__}, transformers={transformers.__version__}')
print(f'CUDA available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    # Quick tensor test
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'CUDA compute OK, result shape={y.shape}')
import llamafactory
print(f'llamafactory OK')
" 2>&1

echo ""
echo "=== Testing spacetools-rl ==="
conda run -n spacetools-rl python -c "
import torch
import transformers
print(f'torch={torch.__version__}, transformers={transformers.__version__}')
print(f'CUDA available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')
if torch.cuda.is_available():
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'CUDA compute OK')
import verl
print(f'verl OK')
import toolshed
print(f'toolshed OK')
import ray
print(f'ray={ray.__version__}')
" 2>&1

echo ""
echo "=== Version alignment check ==="
SFT_TF=$(conda run -n spacetools-sft python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
RL_TF=$(conda run -n spacetools-rl python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
SFT_TORCH=$(conda run -n spacetools-sft python -c "import torch; print(torch.__version__)" 2>/dev/null)
RL_TORCH=$(conda run -n spacetools-rl python -c "import torch; print(torch.__version__)" 2>/dev/null)

echo "SFT:  transformers=$SFT_TF  torch=$SFT_TORCH"
echo "RL:   transformers=$RL_TF  torch=$RL_TORCH"

if [ "$SFT_TF" = "$RL_TF" ]; then
    echo "transformers versions MATCH"
else
    echo "ERROR: transformers versions MISMATCH!"
fi
