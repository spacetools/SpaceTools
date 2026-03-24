# Setup and Troubleshooting

## Environment Setup

### Prerequisites

- Conda (miniconda or anaconda)
- GPU node access (A100 recommended, driver 535+)

### Quick Setup

```bash
# 1. Set up SFT and RL environments
bash setup_envs.sh all   # or: bash setup_envs.sh sft / bash setup_envs.sh rl
bash test_envs.sh        # Verify environments

# 2. Set up per-tool environments
#    The script detects Python + Ray from the active env, so activate spacetools-rl first.
conda activate spacetools-rl
cd SpaceTools-Toolshed
source install_tools/setup_tool_env.sh spacetools-tool-roborefer roborefer
source install_tools/setup_tool_env.sh spacetools-tool-vlm vlm
source install_tools/setup_tool_env.sh spacetools-tool-bbox bbox
source install_tools/setup_tool_env.sh spacetools-tool-graspgen graspgen
```

### Manual Setup

```bash
# SFT
conda create -n spacetools-sft python=3.11 -y
conda activate spacetools-sft
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1
cd SpaceTools-SFT && pip install -e .

# RL (also serves as the base for creating per-tool envs)
conda create -n spacetools-rl python=3.11 -y
conda activate spacetools-rl
pip install "sglang[srt,openai]==0.5.6"  # pulls torch 2.9.1, transformers 4.57.1
pip install "ray[default]==2.47.1"
cd SpaceTools-RL && pip install -e .
cd SpaceTools-Toolshed && pip install -e .
pip install "numpy<2.0.0"  # verl requires numpy<2
pip install cachetools nvidia-cudnn-cu12==9.16.0.29

# flash-attn (must be compiled on a GPU node with CUDA)
# Allocate a GPU node (adjust partition/account for your cluster)
salloc --nodes=1 --gpus=1 --time=2:00:00
conda activate spacetools-rl
export CUDA_HOME=$CONDA_PREFIX
pip install flash-attn --no-build-isolation
pip install nvidia-cudnn-cu12==9.16.0.29  # re-upgrade after flash-attn downgrades it
```

## Configuration

### RL Training Config

Key config flags for multi-turn tool interaction:

```yaml
actor_rollout_ref:
  model:
    freeze_vision_model: true          # Freeze ViT encoder
    freeze_language_model: false       # Train LLM
    freeze_vision_projection: false    # Train projection
  rollout:
    multi_turn:
      enable: true                     # Enable multi-turn tool loop
      max_assistant_turns: 10
      tool_config_path: path/to/tool_config.yaml
```

### LD_LIBRARY_PATH

All training scripts must set `LD_LIBRARY_PATH` for sglang subprocesses:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
```

## Known Issues

- **numpy version conflict**: verl requires `<2.0.0`, toolshed requires `>=2.0.0`.
  Workaround: install `numpy<2.0.0` after toolshed (toolshed works fine with numpy 1.x).

- **cudnn version**: sglang 0.5.6 on torch 2.9.1 requires cudnn 9.15+ (install `nvidia-cudnn-cu12==9.16.0.29`).
  The `flash-attn` pip install may downgrade cudnn — re-upgrade after installing flash-attn.

- **LD_LIBRARY_PATH**: sglang spawns subprocesses that need `libcudart.so.12` and `libcudnn.so`.
  Set `LD_LIBRARY_PATH` to include the conda env's lib directories (see above).

- **GPU allocation with Toolshed**: Each Toolshed tool actor claims 1 GPU. With 2 RoboRefer actors on
  8 GPUs, only 6 remain for training. Set `trainer.n_gpus_per_node=6` and ensure `train_batch_size * n`
  is divisible by 6.

- **torch 2.9.x + Conv3D**: Known PyTorch regression ([#166122](https://github.com/pytorch/pytorch/issues/166122))
  causes slower Conv3D. SpaceTools-SFT patches the error to a warning. May see ~10-20% slowdown in vision encoder.

- **Dataset filter multiprocessing**: `filter_overlong_prompts=True` with `num_proc>1` may silently
  filter all samples. Use `filter_overlong_prompts=False` (all training scripts already do this).

- **SpaceTools-SFT torchrun**: Do NOT use `torchrun -m llamafactory.cli` — LLaMA-Factory internally
  spawns its own `torchrun`, causing double-torchrun conflicts. Use
  `FORCE_TORCHRUN=1 python -m llamafactory.cli train` instead.

- **Multi-node LD_LIBRARY_PATH**: When starting a Ray worker on a remote node via `ssh`, you must
  set `LD_LIBRARY_PATH` on that node too. Ray workers inherit the remote node's environment,
  not the head node's. Without this, sglang subprocesses fail with `libcudart.so.12 not found`.

- **SFT checkpoint config pollution**: SpaceTools-SFT (with transformers 4.57.1) saves extra sections
  in `config.json` and uses `Qwen2VLImageProcessorFast` in `preprocessor_config.json`. Two fixes:
  1. Remove `text_config` from `config.json`: transformers resolves `model_type` to `qwen2_5_vl_text`
     (the text submodel) instead of `qwen2_5_vl`, causing sglang to crash with
     `RuntimeError: Unimplemented model type: qwen2_5_vl_text`.
  2. Replace `preprocessor_config.json` with the base model's version: the Fast processor produces
     different image token counts than the vision encoder expects, causing
     `ValueError: Image features and image tokens do not match` (off-by-one).

  The `run_sft.sh` script applies these fixes automatically.

- **SLURM preemption on shared partitions**: Multi-node jobs on shared partitions may be preempted
  after ~1.5 hours. This appears as `CANCELLED+` in `sacct` with exit code 0:0. Not an OOM or
  code error. Use `--qos=high` or dedicated partitions if available for long-running jobs.
