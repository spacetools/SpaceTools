# SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL (CVPR 2026)

[![Project Page](https://img.shields.io/badge/Project_Page-SpaceTools-blue)](https://spacetools.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.04069)

> Official repository for **SpaceTools** and the **Toolshed** system.  

- ✅ Toolshed system — [installation, demos, and docs](https://github.com/NVlabs/SpaceTools)
- 🔜 Training and evaluation
- 🔜 Data
- 🔜 Pretrained model

---

## 🚀 Highlights

**SpaceTools** empowers VLMs with **vision tools** and **robotic tools** to perform spatial reasoning and real-world manipulation.  
It introduces **Double Interactive Reinforcement Learning (DIRL)**, a two-phase training pipeline for effective multi-tool coordination, and **Toolshed**, a distributed toolkit that enables real-time interaction with compute-heavy multimodal tools during both RL training and inference.

The code release include:

### 🔧 1. Toolshed System [![Released](https://img.shields.io/badge/Released-green)](https://github.com/NVlabs/SpaceTools)
A Ray-based distributed framework for hosting and replicating compute-heavy tools (neural networks, VLMs, code executors) during both training and inference:
- **Included tools**: pointing (RoboRefer, Molmo), depth estimation, SAM2 segmentation, 3D bounding box, grasp generation, code executor, and more
- **Load balancing & queue management**: multiple tool instances with automatic request routing and queuing
- **Environment isolation**: separate conda environments per tool for incompatible dependencies
- **Schema generation**: auto-converts tool docstrings to JSON schemas for LLM frameworks
- **Agentic workflow**: built-in agent with support for OpenAI, Anthropic, Bedrock, and SGLang providers
- **Web UI & dashboard**: interactive agent interface and real-time tool state visualization
- **Multinode deployment**: scale across machines via Ray clusters (SLURM supported)
- **Code execution interface**: Pythonic access to the full toolkit from generated code

### 🧠 **2. Training and Dataset (Coming Soon)**
- DIRL training recipe 
- Toolshed intergrated RL framework
- SFT framework
- SFT + RL dataset 


### 🧠 **3. Model Checkpoint and Evaluation (Coming Soon)**
- Pretrained model checkpoint 
- Spatial benchmark evaluation 



---

## 📝 Citation

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
