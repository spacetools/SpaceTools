# SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL

[![Project Page](https://img.shields.io/badge/Project_Page-SpaceTools-blue)](https://spacetools.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.04069)

> Official repository for **SpaceTools** and the **Toolshed** system (CVPR 2026).

**SpaceTools** empowers VLMs with **vision tools** and **robotic tools** to perform spatial reasoning and real-world manipulation. It introduces **Double Interactive Reinforcement Learning (DIRL)**, a training pipeline with two RL phases and intermediate data collection and SFT, and **Toolshed**, a distributed toolkit that enables real-time tool interaction during both RL training and inference.

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
├── SpaceTools-SFT/               # SFT training (LLaMA-Factory fork)
│   ├── scripts/spacetools/
│   │   ├── run_sft.sh            #   SFT: data prep + training + checkpoint fix
│   │   └── analysis/             #   Dataset balancing utilities
│   └── README.md                 #   SFT documentation
│
├── SpaceTools-RL/                # RL training & evaluation (verl fork)
│   ├── examples/toolshed/
│   │   ├── run_rl_roborefer.sh   #   Step 1: point-tool-only RL (optional)
│   │   ├── run_rl.sh             #   Step 4: full-tool RL (GRPO)
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

The DIRL (Double Interactive RL) pipeline has four steps. The "Double" refers to two RL phases with live tool interaction. Each step produces a checkpoint or data consumed by the next.

```
Step 1: Point-Tool RL       Step 2: Teacher Data       Step 3: SFT             Step 4: Full-Tool RL
(SpaceTools-RL +            Collection                 (SpaceTools-SFT)        (SpaceTools-RL +
 Toolshed)                  (Toolshed + teacher API)                            Toolshed)
                                                                               
Base Qwen2.5-VL-3B                                     Base Qwen2.5-VL-3B      SFT checkpoint
        │                                                      │                       │
        ▼                                                      ▼                       ▼
  RL with detect_one         Teacher VLM (e.g.          Full fine-tune on       GRPO with all
  tool only (GRPO)           Claude) solves tasks       distillation data +     11-17 tools live
        │                    using Toolshed tools       teacher traces          via Toolshed
        ▼                           │                          │                       │
  Distillation data                 ▼                          ▼                       ▼
  (RL traces cleaned         Teacher traces             SFT checkpoint          Final model
   for SFT)                  (multi-tool demos)
```

> **Pre-collected data available.** Steps 1 and 2 produce the training data for SFT (Step 3): Step 1 generates distillation data from point-tool RL traces, and Step 2 collects multi-tool teacher demonstrations via a strong VLM API and Toolshed. We provide all of this pre-collected data in the released SFT dataset ([siyich/spacetools-sft](https://huggingface.co/datasets/siyich/spacetools-sft)), so **most users can skip Steps 1-2 and start directly from Step 3 (SFT)**. Steps 1-2 are documented here for completeness and for users who want to reproduce the full pipeline from scratch.

### Step 1: Point-Tool RL (Optional — pre-collected data provided)

Runs in `SpaceTools-RL/`. Trains the base Qwen2.5-VL-3B model with only the `detect_one` pointing tool (RoboRefer) on RefSpatial and RoboSpatial tasks using GRPO. This teaches the model basic tool-use patterns. The resulting interaction traces are cleaned into distillation data and used as part of the SFT training data in Step 3.

Since we release the pre-collected distillation data as part of the SFT dataset, this step is optional. Run it only if you want to regenerate the data yourself.

```bash
conda activate spacetools-rl
cd SpaceTools-RL

ROBOREFER_MODEL=/path/to/RoboRefer-8B-SFT \
    bash examples/toolshed/run_rl_roborefer.sh
```

Output: `experiments/rl_roborefer_output/global_step_XXX/`

Expected time: ~10-15 hours on 8x A100 (15 epochs). The script supports automatic resume — rerun the same command to continue from the latest checkpoint. See [SpaceTools-RL/README.md](SpaceTools-RL/README.md) for details.

### Step 2: Teacher Data Collection (Optional — pre-collected data provided)

Uses a strong teacher VLM (e.g., Claude) together with Toolshed to solve spatial reasoning tasks with the full tool suite. The teacher interacts with tools in real time to produce high-quality multi-tool demonstration traces covering diverse tool combinations (depth estimation, bounding boxes, visual grounding, VLM reasoning, etc.). These traces are cleaned and combined with the distillation data from Step 1 to form the SFT dataset.

Since we release the pre-collected teacher traces as part of the SFT dataset, this step is optional. The data preparation scripts in `SpaceTools-SFT/scripts/spacetools/data_prep/` show how the raw traces are processed and balanced.

### Step 3: SFT (Supervised Fine-Tuning)

Runs in `SpaceTools-SFT/`. Downloads the SFT dataset from HuggingFace (which includes both the point-tool RL distillation data from Step 1 and the teacher traces from Step 2), injects tool schemas into the system prompt, trains a full fine-tune on Qwen2.5-VL-3B, and fixes the checkpoint.

```bash
conda activate spacetools-sft
cd SpaceTools-SFT

# V1: reasoning only (11 tools, ~7000 samples)
VERSION=v1 TOOL_CONFIG=../SpaceTools-RL/examples/toolshed/toolshed_v1_config.yaml \
    bash scripts/spacetools/run_sft.sh

# V2: full (17 tools including robot, ~7900 samples)
VERSION=v2 TOOL_CONFIG=../SpaceTools-RL/examples/toolshed/toolshed_v2_config.yaml \
    bash scripts/spacetools/run_sft.sh
```

Output: `experiments/sft_<version>_<timestamp>/sft_checkpoint/`

Expected time: ~3-4 hours on 8x A100 (3000 steps). See [SpaceTools-SFT/README.md](SpaceTools-SFT/README.md) for details.

### Step 4: Full-Tool RL (Reinforcement Learning with Full Tool Interaction)

Runs in `SpaceTools-RL/`. Takes the SFT checkpoint from Step 3, starts a Ray cluster with Toolshed tool actors, and runs GRPO training with multi-turn execution of all tools during rollouts.

```bash
conda activate spacetools-rl
cd SpaceTools-RL

SFT_CHECKPOINT=/path/to/sft_checkpoint \
ROBOREFER_MODEL=/path/to/RoboRefer-8B-SFT \
DEPTH_CHECKPOINT=/path/to/depth_pro.pt \
    bash examples/toolshed/run_rl.sh
```

Output: `experiments/rl_<version>_<timestamp>/rl_output/global_step_XXX/`

Expected time: ~8-12 hours on 8x A100 (1 epoch, ~60 steps). See [SpaceTools-RL/README.md](SpaceTools-RL/README.md) for details.

---

## Evaluation

Runs in `SpaceTools-RL/`. Evaluates a model checkpoint on all 9 paper benchmarks with live tool execution. You can evaluate either your own trained checkpoint (from the pipeline above) or the pretrained checkpoint once it is released.

> **Note:** The pretrained model checkpoint is not yet publicly available. To reproduce the paper results now, run Steps 3-4 of the pipeline above (SFT → Full-Tool RL) to produce your own checkpoint, then evaluate it.

```bash
conda activate spacetools-rl
cd SpaceTools-RL

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

| Dataset | Samples | Used in | Description |
|---|---|---|---|
| [spacetools-rlpointtools](https://huggingface.co/datasets/siyich/spacetools-rlpointtools) | ~4000 | Step 1 (Point-Tool RL) | RL prompts for point/refer tools only |
| [spacetools-sft](https://huggingface.co/datasets/siyich/spacetools-sft) | ~7900 | Step 3 (SFT) | SFT data in ShareGPT format with images; includes pre-collected distillation data from Step 1 and teacher traces from Step 2 |
| [spacetools-rlfulltools](https://huggingface.co/datasets/siyich/spacetools-rlfulltools) | ~5500 | Step 4 (Full-Tool RL) | RL prompts for full tool suite |
| [spacetools-eval-benchmarks](https://huggingface.co/datasets/siyich/spacetools-eval-benchmarks) | — | Evaluation | Evaluation benchmark parquets |

---

## Architecture

### Double Interactive RL (DIRL)

The "Double" in DIRL refers to two RL phases with live tool interaction:

```
Step 1: Point-Tool RL (SpaceTools-RL + Toolshed)
  └── GRPO on base Qwen2.5-VL-3B with detect_one (pointing) tool only
       └── Teaches basic tool-use patterns on RefSpatial + RoboSpatial tasks
       └── Produces RL traces → cleaned into distillation data for SFT
       └── (Pre-collected data released; this step is optional)

Step 2: Teacher Data Collection (Toolshed + teacher API)
  └── Strong teacher VLM (e.g., Claude) solves tasks with full Toolshed tools
       └── Produces multi-tool demonstration traces
       └── (Pre-collected traces released; this step is optional)

Step 3: SFT (SpaceTools-SFT)
  └── Full fine-tune Qwen2.5-VL-3B on distillation data (Step 1) + teacher traces (Step 2)
       └── Vision tower frozen, language model trained
       └── Tool schemas injected into system prompt

Step 4: Full-Tool RL (SpaceTools-RL + Toolshed)
  └── GRPO on SFT checkpoint with all 11-17 tools live
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
| `spacetools-sft` | SFT training (SpaceTools-SFT, deepspeed) |
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
