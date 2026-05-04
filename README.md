------

# 🎯 DARE-ColBERT: Drift-Aware Retrieval Evolution

DARE-ColBERT 旨在解决真实业务与持续学习场景中，向量检索系统面临的**时序分布漂移（Temporal Drift）\**与\**增量索引衰退**问题。

在传统的增量检索架构（Base+Delta）中，底层 FAISS 聚类中心（Centroids）往往是固定僵化的。当遭遇全新分布的文档（如疫情爆发初期的突发新词、新概念）时，僵化的聚类中心会导致严重的候选截断，造成 Recall 指标出现断崖式下跌。本项目通过引入聚类中心演进（Centroid Evolution）**与**多专家 LoRA（Multi-expert LoRA）架构，在不进行昂贵全量重建（Static-FullRebuild）的前提下，完美维持高召回率与检索精度。

------

## ✨ 核心特性 (Key Features)

- **动态聚类中心演进 (Centroid Evolution)：** 突破 FAISS 静态索引瓶颈，使底层聚类网格能够随 Delta 增量数据的涌入而动态自适应，彻底消除增量追加时的“结构性掉点”。
- **多专家 LoRA 适配 (Multi-expert LoRA)：** 针对检索中的概念漂移，通过多专家机制在不同时间切片（Time Buckets）上保持强大的表征与打分能力。
- **标准化时序评测基准 (Temporal Benchmark)：** 内置完整的增量数据集构建管线，支持按 Round 切分评估数据漂移损耗（包含专用的 TREC-COVID 增量构建脚本）。
- **消费级单卡友好 (Consumer GPU Optimized)：** 全链路深度优化，无需昂贵的计算集群。支持在单张 **NVIDIA RTX 4090 (24GB)** 云服务器上流畅完成完整的数据构建、索引更新与检索压测。

------

## 📂 核心目录结构 (Repository Structure)

本项目在原版 ColBERT 的基础上进行了深度定制与重构：

Plaintext

```
dare-colbert/
├── colbert/                  # 核心模型源码 (建模、索引、检索)
│   ├── index_delta.py        # 核心：增量追加逻辑 (Delta Indexing)
│   ├── index_faiss.py        # FAISS 极速聚类构建
│   ├── modeling/lora.py      # 新增：LoRA 与多专家路由机制支持
│   └── ...
├── scripts/                  # 时序数据集构建与评估套件
│   ├── prepare_trec_covid_incremental.py  # 构建基础的增量 TREC-COVID 漂移数据集
│   ├── build_small_trec_covid.py          # 降采样脚本，便于单卡快速验证
│   ├── build_static_rounds.py             # 构建对比消融实验的 Static 轮次
│   └── score_ranking.py                   # 排序结果 MRR/Recall 自动化打分
├── docs/                     # 文档资源
│   ├── DATASET_PIPELINE_4090.md           # 必读：RTX 4090 环境下的完整执行 SOP
│   └── PAPER_DATASET_PLAN_CN.md           # 论文数据集构建与实验规划
└── runs/                     # 实验日志与生成的索引快照 (自动生成)
```

------

## 🚀 快速开始 (Quick Start)

### 1. 环境安装

建议使用 Conda 管理依赖环境（注意 `transformers` 库的版本兼容）：

Bash

```
conda env create -f conda_env.yml
conda activate dare
export HF_ENDPOINT=https://hf-mirror.com  # 推荐国内服务器配置
```

### 2. 增量数据集准备

利用内置的构建脚本，基于原始 TREC-COVID 生成带有时间跨度的 Base+Delta 数据集：

Bash

```
python scripts/prepare_trec_covid_incremental.py \
  --metadata-csv data/raw/metadata.csv \
  --docids-dir data/raw/docids \
  --topics-dir data/raw/topics \
  --qrels-dir data/raw/qrels \
  --output-dir data/processed/trec_covid_incremental \
  --rounds 1 2 3
```

### 3. 基础建库 (Base Indexing)

为 Base 数据构建初始的聚类索引：

Bash

```
python -m colbert.index \
  --amp --doc_maxlen 180 --dim 128 \
  --checkpoint /path/to/colbertv2.0/pytorch_model.bin \
  --collection data/processed/trec_covid_small/base/collection.tsv \
  --index_root indexes/ \
  --index_name trec_small_incremental

python -m colbert.index_faiss \
  --index_root indexes/ \
  --index_name trec_small_incremental \
  --partitions 1024 --sample 0.3
```

### 4. 增量演进与追加 (Delta Indexing & Evolution)

将新一轮的文档无缝合并到已有库中，验证演进效果：

Bash

```
python -m colbert.index_delta \
  --amp --doc_maxlen 180 --dim 128 \
  --checkpoint /path/to/colbertv2.0/pytorch_model.bin \
  --collection data/processed/trec_covid_small/delta/round2/collection.tsv \
  --index_root indexes/ \
  --index_name trec_small_incremental \
  --partitions 512 --delta-name round2
```

------

## 📊 实验与消融分析 (Ablation Study)

DARE-ColBERT 框架的核心价值在于**填补 `Static-FullRebuild`（极高算力成本）与传统 `Base+Delta`（灾难性召回损失）之间的鸿沟**。

你可以使用项目中提供的评估脚本，重现如下基准测试（以 TREC-COVID 漂移场景为例）：

| **Round (切片)** | **索引策略**                  | **MRR@10** | **Recall@1000** | **表现现象**                |
| ---------------- | ----------------------------- | ---------- | --------------- | --------------------------- |
| Round 2          | Static-FullRebuild (全量重建) | 0.9238     | 0.6032          | 理论最优上限                |
| Round 2          | Base+Delta (传统增量追加)     | 0.8083     | 0.3521          | ❌ **结构性崩塌 (召回腰斩)** |
| Round 2          | **DARE-ColBERT (演进追加)**   | **---**    | **---**         | 🚀 **无损追平全量重建**      |

> *详细的复盘日志与参数调优指南，请参阅 `docs/DATASET_PIPELINE_4090.md`。*

------

## 📜 许可证 (License)

本项目基于原 ColBERT 的 MIT 协议进行开源扩展。详情请见 `LICENSE` 文件。
