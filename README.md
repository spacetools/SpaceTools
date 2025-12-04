# SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL

> Official repository for **SpaceTools** and the **Toolshed** system.  
> 🛠️ **Code coming soon** — we are preparing a full release including the Toolshed infrastructure, DIRL training pipeline, and evaluation scripts.

---

## 📌 Overview

**SpaceTools** is a framework that empowers VLMs with **vision tools** and **robotic tools** for spatial reasoning and real-world manipulation.  
It introduces **Double Interactive Reinforcement Learning (DIRL)**, a two-phase training pipeline that enables effective multi-tool coordination.

The code release will include both:

### 🔧 **1. Toolshed (System Release)**
A scalable infrastructure for deploying compute-heavy tools during both training and inference:
- Isolated environments for each tool  
- Decoupled resource scaling  
- Async parallel workers per tool  
- Support for heavy tools (segmentation, pointing, depth, 3D box, grasp prediction)  

### 🧠 **2. SpaceTools (Model + Training Release)**
- DIRL training pipeline  
- SFT + RL dataset  
- Tool-augmented inference  
- Spatial benchmark evaluation  

For project details and demos:  
👉 **Project Page:** https://spacetools.github.io/  
👉 **Paper:** https://arxiv.org/pdf/2512.04069



---

## 📦 Installation (coming soon)

We will provide detailed setup instructions, including:

- Recommended environment (conda / pip)  
- CUDA / PyTorch version requirements  
- Setup for Toolshed workers and servers  
- Integration of Toolshed for interactive RL
- Integration of Toolshed for zero-shot frontier model reasoning
- Supervise fine-tuning for tool use
- Dependencies for VLMs, RL, SFT, and each tool backend

```bash
# Placeholder – installation instructions coming soon
git clone https://github.com/spacetools/spacetools.git
cd spacetools
