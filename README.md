# SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL

[![Project Page](https://img.shields.io/badge/Project_Page-SpaceTools-blue)](https://spacetools.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.04069)

> Official repository for **SpaceTools** and the **Toolshed** system (CVPR 2026).

**SpaceTools** empowers VLMs with **vision tools** and **robotic tools** to perform spatial reasoning and real-world manipulation. It introduces **Double Interactive Reinforcement Learning (DIRL)**, a two-phase training pipeline, and **Toolshed**, a distributed toolkit that enables real-time tool interaction during both RL training and inference.

---

## Release

| Component | Link |
|---|---|
| Toolshed (distributed tool serving) | [NVlabs/SpaceTools-Toolshed](https://github.com/NVlabs/SpaceTools-Toolshed) |
| SFT training (LLaMA-Factory fork) | [ChicyChen/SpaceTools-SFT](https://github.com/ChicyChen/SpaceTools-SFT) |
| RL training & eval (verl fork) | [ChicyChen/SpaceTools-RL](https://github.com/ChicyChen/SpaceTools-RL) |
| SFT dataset | [siyich/spacetools-sft](https://huggingface.co/datasets/siyich/spacetools-sft) |
| RL dataset (full tools) | [siyich/spacetools-rlfulltools](https://huggingface.co/datasets/siyich/spacetools-rlfulltools) |
| RL dataset (point tools) | [siyich/spacetools-rlpointtools](https://huggingface.co/datasets/siyich/spacetools-rlpointtools) |
| Evaluation benchmarks | [siyich/spacetools-eval-benchmarks](https://huggingface.co/datasets/siyich/spacetools-eval-benchmarks) |
| Pretrained model checkpoint | Coming soon — follow this repo for updates |

---

## Repository Structure

```
SpaceTools/
├── SpaceTools-Toolshed/          # Distributed tool serving (Ray-based)
│   ├── toolshed/                 #   Core framework
│   ├── install_tools/            #   Per-tool conda env setup
│   └── README.md                 #   Full Toolshed documentation
│
├── LLaMA-Factory/                # SFT training
│   ├── scripts/spacetools/
│   │   ├── run_sft.sh            #   SFT: data prep + training + checkpoint fix
│   │   └── analysis/             #   Dataset balancing utilities
│   └── README.md                 #   SFT documentation
│
├── verl/                         # RL training & evaluation
│   ├── examples/toolshed/
│   │   ├── run_rl.sh             #   RL: GRPO with live tool execution
│   │   ├── run_rl_roborefer.sh   #   RL: roborefer-only (standalone)
│   │   ├── run_eval.sh           #   Evaluation on paper benchmarks
│   │   ├── toolshed_v1_config.yaml  # 11 tools (reasoning only)
│   │   └── toolshed_v2_config.yaml  # 17 tools (with robot)
│   ├── analysis/                 #   Result analysis scripts
│   └── README.md                 #   RL & eval documentation
│
├── docs/SETUP.md                 # Environment setup & troubleshooting
├── setup_envs.sh                 # Create conda environments
├── install_flash_attn.sh         # Flash attention compilation
└── test_envs.sh                  # Verify environments
```

Each sub-directory is a git submodule with its own upstream tracking.

> **Note on upstream integration:** SpaceTools-SFT and SpaceTools-RL are forks of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [verl](https://github.com/volcengine/verl) with minimal, targeted modifications. We are actively working to upstream our Toolshed-based multi-turn tool training into the latest versions of both frameworks. The goal is to eliminate the need for custom forks entirely, so that users can install the official releases and run SpaceTools training out of the box. Until then, the forks are kept as close to upstream as possible to make rebasing straightforward.

---

## Prerequisites

- **Hardware**: 8x A100 (80GB) GPUs recommended. Minimum 8x GPUs with 40GB+ VRAM each. The tool actors consume ~2 GPUs, leaving 6 for training.
- **Software**: Linux, CUDA 12.x, conda (miniconda or anaconda)
- **Storage**: ~50GB for conda environments, ~20GB for datasets (downloaded automatically), ~30GB for model checkpoints

---

## Quick Start

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/spacetools/SpaceTools.git
cd SpaceTools
```

### 2. Set Up Environments

```bash
# Create SFT and RL conda environments
bash setup_envs.sh all
bash test_envs.sh

# Create per-tool environments (from spacetools-rl)
conda activate spacetools-rl
cd SpaceTools-Toolshed
source install_tools/setup_tool_env.sh spacetools-tool-roborefer roborefer
source install_tools/setup_tool_env.sh spacetools-tool-vlm vlm
source install_tools/setup_tool_env.sh spacetools-tool-bbox bbox
source install_tools/setup_tool_env.sh spacetools-tool-graspgen graspgen
cd ..
```

See [docs/SETUP.md](docs/SETUP.md) for detailed setup, manual installation, and troubleshooting.

### 3. Download Required Models

Training datasets are downloaded automatically from HuggingFace by the scripts. You need to manually download two tool model checkpoints used by the Toolshed tool actors during RL training and evaluation:

**RoboRefer-8B-SFT** — visual grounding model ([Zhoues/RoboRefer-8B-SFT](https://huggingface.co/Zhoues/RoboRefer-8B-SFT)):
```bash
# Option A: huggingface-cli
huggingface-cli download Zhoues/RoboRefer-8B-SFT --local-dir models/RoboRefer-8B-SFT

# Option B: git lfs
git lfs install
git clone https://huggingface.co/Zhoues/RoboRefer-8B-SFT models/RoboRefer-8B-SFT
```

**depth_pro.pt** — monocular depth estimation checkpoint ([Apple ml-depth-pro](https://github.com/apple/ml-depth-pro)):
```bash
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P models/
```

Set these as environment variables (used by RL and eval scripts):
```bash
export ROBOREFER_MODEL=models/RoboRefer-8B-SFT
export DEPTH_CHECKPOINT=models/depth_pro.pt
```

> **Note:** These models are only needed for RL training and evaluation. SFT training does not require them.

---

## Training Pipeline

The DIRL pipeline has two phases. Each step produces a checkpoint consumed by the next.

### Phase 1: SFT (Supervised Fine-Tuning)

Runs in `LLaMA-Factory/`. Downloads SFT data from HuggingFace, injects tool schemas, trains a full fine-tune on Qwen2.5-VL-3B, and fixes the checkpoint.

```bash
conda activate spacetools-sft
cd LLaMA-Factory

# V1: reasoning only (11 tools, ~7000 samples)
VERSION=v1 TOOL_CONFIG=../verl/examples/toolshed/toolshed_v1_config.yaml \
    bash scripts/spacetools/run_sft.sh

# V2: full (17 tools including robot, ~7900 samples)
VERSION=v2 TOOL_CONFIG=../verl/examples/toolshed/toolshed_v2_config.yaml \
    bash scripts/spacetools/run_sft.sh
```

Output: `experiments/sft_<version>_<timestamp>/sft_checkpoint/`

Expected time: ~3-4 hours on 8x A100 (3000 steps). See [LLaMA-Factory/README.md](LLaMA-Factory/README.md) for details.

### Phase 2: RL (Reinforcement Learning with Tool Interaction)

Runs in `verl/`. Starts a Ray cluster with Toolshed tool actors, then runs GRPO training with multi-turn tool execution during rollouts.

```bash
conda activate spacetools-rl
cd verl

SFT_CHECKPOINT=/path/to/sft_checkpoint \
ROBOREFER_MODEL=/path/to/RoboRefer-8B-SFT \
DEPTH_CHECKPOINT=/path/to/depth_pro.pt \
    bash examples/toolshed/run_rl.sh
```

Output: `experiments/rl_<version>_<timestamp>/rl_output/global_step_XXX/`

Expected time: ~8-12 hours on 8x A100 (1 epoch, ~60 steps). See [verl/README.md](verl/README.md) for details.

---

## Evaluation

Runs in `verl/`. Evaluates a model checkpoint on all 9 paper benchmarks with live tool execution. You can evaluate either your own trained checkpoint (from the pipeline above) or the pretrained checkpoint once it is released.

> **Note:** The pretrained model checkpoint is not yet publicly available. To reproduce the paper results now, run the full SFT → RL pipeline above to produce your own checkpoint, then evaluate it.

```bash
conda activate spacetools-rl
cd verl

ROBOREFER_MODEL=/path/to/RoboRefer-8B-SFT \
DEPTH_CHECKPOINT=/path/to/depth_pro.pt \
    bash examples/toolshed/run_eval.sh /path/to/model_checkpoint
```

Run specific benchmarks:
```bash
bash examples/toolshed/run_eval.sh /path/to/model robospatial bopgrasp blinkdepth
```

| Benchmark key | Paper metric |
|---|---|
| `robospatial` | RoboSpatial (VQA, Vacant, Overall) |
| `reflocation`, `refplacement`, `refunseen` | RefSpatial (averaged) |
| `blinkdepth` | BLINK Relative Depth |
| `cvb2drelation` | CVBench 2D Relation |
| `cvb3ddepth` | CVBench 3D Depth |
| `boppose` | BOP-ask Pose |
| `bopgrasp` | BOP-ask Grasp (MACE + SR) |

---

## Datasets

All datasets are hosted on HuggingFace and downloaded automatically by the training scripts.

| Dataset | Samples | Description |
|---|---|---|
| [spacetools-sft](https://huggingface.co/datasets/siyich/spacetools-sft) | ~7900 | SFT data in ShareGPT format with images |
| [spacetools-rlfulltools](https://huggingface.co/datasets/siyich/spacetools-rlfulltools) | ~5500 | RL prompts for full tool suite |
| [spacetools-rlpointtools](https://huggingface.co/datasets/siyich/spacetools-rlpointtools) | ~4000 | RL prompts for point/refer tools only |
| [spacetools-eval-benchmarks](https://huggingface.co/datasets/siyich/spacetools-eval-benchmarks) | — | Evaluation benchmark parquets |

---

## Architecture

### Double Interactive RL (DIRL)

```
Phase 1: SFT (LLaMA-Factory)
  └── Full fine-tune Qwen2.5-VL-3B with tool-augmented conversations
       └── Vision tower frozen, language model trained
       └── Tool schemas injected into system prompt

Phase 2: RL (verl + Toolshed)
  └── GRPO training with multi-turn tool interaction
       └── AgentLoopManager → ToolAgentLoop
            ├── Prompt → sglang inference → assistant response
            ├── Parse tool calls (hermes format)
            ├── Execute tools via Toolshed (Ray actors)
            ├── Append tool response → next turn
            └── Repeat until answer or max turns
```

### Toolshed

A Ray-based distributed framework for hosting compute-heavy tools during training and inference:
- **7-8 tool types**: RoboRefer, Molmo VLM, SAM2, depth estimator, bounding box, grasp generator, vision ops, (mock robot)
- **Environment isolation**: Each tool runs in its own conda environment
- **Load balancing**: Multiple instances with automatic request routing
- **GPU sharing**: Tools share GPUs with fractional allocation

### Conda Environments

| Environment | Purpose |
|---|---|
| `spacetools-sft` | SFT training (LLaMA-Factory, deepspeed) |
| `spacetools-rl` | RL training, evaluation, tool orchestration |
| `spacetools-tool-roborefer` | RoboRefer tool actor |
| `spacetools-tool-vlm` | VLM + SAM2 + depth tool actors |
| `spacetools-tool-bbox` | Bounding box + vision ops tool actors |
| `spacetools-tool-graspgen` | Grasp generation tool actor |

All environments share Python 3.11 and Ray 2.47.1 for cross-env compatibility.

---

## Citation

```bibtex
@misc{chen2025spacetoolstoolaugmentedspatialreasoning,
    title={SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL},
    author={Siyi Chen and Mikaela Angelina Uy and Chan Hee Song and Faisal Ladhak and Adithyavairavan Murali and Qing Qu and Stan Birchfield and Valts Blukis and Jonathan Tremblay},
    year={2025},
    eprint={2512.04069},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2512.04069}
}
```
