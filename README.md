# **Deep Convolutional Critic (DCC)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)

This repository contains the official PyTorch implementation for the paper: **"Beyond Reconstruction: A Deep Convolutional Critic with Spectral Normalization for Superior Bearing Fault Anomaly Detection"**.

我们挑战了基于重构的异常检测方法的主导地位，提出了一种更直接、更鲁棒、性能更优越的无监督异常检测新范式。该方法的核心是一个经过**谱归一化（Spectral Normalization）**正则化的**深度卷积批判家（Deep Convolutional Critic, DCC）**模型。

**我们的模型在三个主流公开轴承故障数据集（CWRU, SEU, MFPT）上均取得了完美的AUC（1.0000），显著优于所有基准模型。**

## 摘要 (Abstract)

Unsupervised anomaly detection is paramount for predictive maintenance. While reconstruction-based models like Autoencoders (AEs) are prevalent, they often involve significant architectural complexity and may not achieve optimal performance. We propose a novel, non-reconstructive framework centered on a **Deep Convolutional Critic (DCC)**, a 1D CNN regularized with **Spectral Normalization**. The DCC directly learns a "normality score" and demonstrated exceptional performance, achieving a perfect AUC of 1.0000 across the CWRU, SEU, and MFPT datasets, significantly outperforming all baselines. Our work proves that a direct, well-regularized scoring approach is a more potent and robust strategy for bearing fault detection.

## 核心思想：DCC 框架 (Key Idea: The DCC Framework)

我们的框架遵循一个简单而高效的“解耦-评分”流程：

1.  **信号分段与逐段归一化**: 将原始信号切片，并对每个片段独立进行Z-score归一化，以解耦信号的结构与能量信息，迫使模型关注内在模式。
2.  **DCC评分**: 将归一化后的片段输入到一个轻量级的、经过谱归一化正则的1D CNN（即DCC）中，直接输出一个标量“正常度分数”。
3.  **训练与推理**: 模型仅在正常数据上训练，目标是最大化正常样本的得分。在推理时，低分表示异常。

![Figure 2: Illustration of Per-Segment Normalization](https://tc.z.wiki/autoupload/f/3tdjW0cTpp7UnyEeZ7pj89DBO7i3hlAO9Eehlq6b2wuyl5f0KlZfm6UsKj-HyTuv/20250707/rKIM/950X772/wechat_2025-07-07_180651_754.png)
> **图**: 所提出的DCC异常检测框架流程图。 

## 主要亮点 (Highlights)
- **范式创新**: 首次提出一种非重构的“直接评分”模型（DCC），替代复杂的自编码器架构。
- **技术融合**: 将GAN领域的谱归一化技术成功引入单分类故障诊断任务，确保训练稳定性和评分函数的平滑性。
- **卓越性能**: 在三个公开数据集上均达到**完美的AUC（1.0000）**，证明了其强大的判别能力。
- **鲁棒且通用**: 在不同数据集和含噪环境下均保持顶尖性能，展示了卓越的泛化能力和工业应用潜力。
- **轻量高效**: 模型结构简单，训练快速，推理仅需一次前向传播，适合实时在线监测。

## 仓库结构 (Repository Structure)
```
.
├── dataset1.py          # MFPTDataset
├── dataset2.py          # SEUDataset
├── dataset3.py          # CWRUDataset
├── main.py             # 主执行脚本 (训练与评估)
└── README.md
```

## 快速开始 (Getting Started)

### 1. 克隆仓库
```bash
git clone https://github.com/ZS520L/Deep-Convolutional-Critic.git
cd Deep-Convolutional-Critic
```

### 2. 配置环境
建议使用`conda`或`venv`创建虚拟环境，然后安装所需依赖。
```bash
# 使用 conda (推荐)
conda create -n dcc python=3.9
conda activate dcc

# 主要依赖包括: pytorch, numpy, scikit-learn, matplotlib, tqdm
```

### 3. 数据准备
请从以下链接下载数据集，并将其解压到 `data/` 目录下，保持与仓库结构一致。
- **CWRU Dataset:** [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)
- **SEU Dataset:** [Southeast University Gearbox Data](https://github.com/cathysiyu/Mechanical-datasets) (该链接包含多个数据集，请找到SEU轴承部分)
- **MFPT Dataset:** [Machinery Failure Prevention Technology Society](https://mfpt.org/fault-data-sets/)


## 运行实验 (Running the Experiments)

我们通过 `main.py` 脚本来执行所有实验。

### 训练DCC模型、评估与复现结果
使用以下命令加载已训练好的模型进行评估，并复现论文中的AUC、t-SNE图和分数分布图。

```bash
# 评估在 三个数据集 上的所有模型效果
python main.py
```
> 评估脚本将会在控制台打印AUC分数。

## 主要结果 (Main Results)

### 定量性能对比 (AUC)
我们的DCC模型在所有数据集上均优于所有基准模型。

| Method             | CWRU       | SEU        | MFPT       |
| :-----------------: | :---------: | :---------: | :---------: |
| DCC                |   1.0000   |   1.0000   |   1.0000   |
| ConvAE             |   0.9801   |   1.0000   |   0.7078   |
| DeepSVDD           |   0.9996   |   1.0000   |   0.9381   |
| iForest            |   0.4371   |   0.3314   |   0.2947   |
| VAE                |   1.0000   |   1.0000   |   0.9780   |
| LSTMPredictor      |   0.7915   |   0.9740   |   0.5090   |
| DCGAN              |   0.5809   |   0.5690   |   0.5013   |
| WGAN-GP            |   0.6102   |   0.8555   |   0.5303   |


### 可视化分析
DCC学习到了一个高度可分的特征空间，正常与异常样本被完美分离。

![Figure 5: t-SNE Visualization](https://tc.z.wiki/autoupload/f/3tdjW0cTpp7UnyEeZ7pj89DBO7i3hlAO9Eehlq6b2wuyl5f0KlZfm6UsKj-HyTuv/20250707/Txxh/1214X414/wechat_2025-07-07_194811_122.png)
> **图**: 三个数据集上学习到的特征的t-SNE可视化。

![Figure 6. Visualization of the DCC's Discriminative Power and Resulting Performance.](https://tc.z.wiki/autoupload/f/3tdjW0cTpp7UnyEeZ7pj89DBO7i3hlAO9Eehlq6b2wuyl5f0KlZfm6UsKj-HyTuv/20250707/cA0w/924X745/wechat_2025-07-07_194822_083.png)
> **图**: DCC模型输出的正常/异常分数分布直方图与ROC曲线。

## 许可证 (License)
本仓库遵循 [MIT License](LICENSE)。
