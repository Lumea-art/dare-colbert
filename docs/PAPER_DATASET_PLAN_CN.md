# 单卡 4090 论文数据方案

这份方案面向你的当前目标：在单张 RTX 4090 上，完成一个“原始 ColBERTv1 基线 + 面向漂移/增量检索改造版”的可复现实验。

核心原则只有三条：

1. **基线必须标准**：原始 ColBERTv1 不要和你的 MoE/LoRA 改造混在一起。
2. **漂移必须真实**：不要用静态切分假装时序漂移，优先用天然分轮次增长的数据。
3. **单卡必须跑得动**：文档侧冻结底座、离线编码、增量索引，避免全量反复重建。

---

## 1. 推荐的数据组合

### 1.1 原始基线：MS MARCO Passage Ranking

用途：

- 训练并验证**原始 ColBERTv1**；
- 作为你论文里的“标准强基线”；
- 也可以作为你改造模型的第一阶段通用检索预训练来源。

为什么保留它：

- ColBERT 系列最常见的监督训练基线就是 MS MARCO；
- 审稿人更容易接受；
- 你的改造如果完全绕开 MS MARCO，论文说服力会弱很多。

### 1.2 漂移 + 增量评测：TREC-COVID / CORD-19

用途：

- 构建 base + delta 的增量索引实验；
- 检验你的 routing 是否对时序变化、来源变化、新增主题有帮助；
- 验证“训练时路由”和“索引/检索时路由”是否一致。

为什么它适合你的系统：

- 数据本身按 round 演化；
- 每轮有新的文档集合、topic、qrels；
- 很适合你现在已经做好的 `base + delta` 检索结构；
- 以 `title + abstract` 作为一个检索单元时，单卡 4090 可以接受。

---

## 2. 不建议你第一篇就上的数据

### 2.1 不建议只做 BEIR

原因：

- BEIR 强在零样本泛化，但不天然支持“增量索引”；
- 你要论证的是漂移/增量路由，不是单纯 domain generalization；
- 只做 BEIR，论文会更像“泛化检索”而不是“漂移检索”。

### 2.2 不建议直接做全量 CORD-19 全文 passage 化

原因：

- 计算量和索引量会明显增加；
- 文档切 passage 后，qrels 映射也更复杂；
- 对第一篇论文而言，`title + abstract` 已足够构建一个 defensible 的 MVP。

### 2.3 不建议一开始自造合成漂移数据

原因：

- 审稿人通常更相信真实 evolving benchmark；
- 合成漂移更适合做补充实验，不适合做主实验。

---

## 3. 当前仓库里已经准备好的数据脚本

你现在可以直接使用下面这些脚本：

- `scripts/prepare_trec_covid_incremental.py`
- `scripts/simple_bm25_topk.py`
- `scripts/annotate_ranking_with_qrels.py`
- `scripts/score_ranking.py`
- `docs/DATASET_PIPELINE_4090.md`

这套脚本解决的是：

- 把 TREC-COVID / CORD-19 原始文件整理成 ColBERT 可读的 `collection.tsv`
- 生成 `collection_meta.jsonl`
- 按 round 生成 `queries.tsv` 和 `qrels.tsv`
- 生成 base / delta 数据目录
- 对检索结果做基础指标打分

另外我已经补了一个关键开关：

- `--docids-mode cumulative`
- `--docids-mode incremental`

含义是：

- 如果你的 `docids-rndX.txt` 本身就是“截至第 X 轮的累计全集”，用 `cumulative`
- 如果你的 `docids-rndX.txt` 只包含“这一轮新增文档”，用 `incremental`

这一步很重要。因为如果 round 文件语义理解错了，你后面的增量实验会直接失真。

---

## 4. 推荐的目录组织

建议你按下面的目录组织原始数据：

```text
data/
  raw/
    msmarco/
      collection.tsv
      queries.train.tsv
      triples.train.jsonl
      qrels.dev.tsv
    cord19/
      metadata.csv
    trec-covid/
      docids/
        docids-rnd1.txt
        docids-rnd2.txt
        ...
      topics/
        topics-rnd1.xml
        topics-rnd2.xml
        ...
      qrels/
        qrels-covid_d5_j0.5-rnd1.txt
        qrels-covid_d5_j0.5-rnd2.txt
        ...
```

处理后的数据建议放在：

```text
data/
  processed/
    trec_covid/
      base/
      delta/
      rounds/
      manifest.json
      pid_lookup.jsonl
```

---

## 5. TREC-COVID 数据处理命令

### 5.1 标准累计模式

如果每个 `docids-rndX.txt` 都已经包含该轮可见的完整文档集合：

```bash
python scripts/prepare_trec_covid_incremental.py \
  --metadata-csv data/raw/cord19/metadata.csv \
  --docids-dir data/raw/trec-covid/docids \
  --topics-dir data/raw/trec-covid/topics \
  --qrels-dir data/raw/trec-covid/qrels \
  --output-dir data/processed/trec_covid \
  --rounds 1 2 3 4 5 \
  --docids-mode cumulative \
  --topic-field query+question
```

### 5.2 增量模式

如果每个 `docids-rndX.txt` 只包含该轮新增文档：

```bash
python scripts/prepare_trec_covid_incremental.py \
  --metadata-csv data/raw/cord19/metadata.csv \
  --docids-dir data/raw/trec-covid/docids \
  --topics-dir data/raw/trec-covid/topics \
  --qrels-dir data/raw/trec-covid/qrels \
  --output-dir data/processed/trec_covid \
  --rounds 1 2 3 4 5 \
  --docids-mode incremental \
  --topic-field query+question
```

Windows PowerShell 下，如果你显式传 `topics-pattern` / `qrels-pattern`，要这样写：

```powershell
--topics-pattern 'topics-rnd{round}.xml'
--qrels-pattern 'qrels-rnd{round}.txt'
```

否则 `{round}` 可能被 PowerShell 吃掉。

---

## 6. 处理后产物如何使用

### 6.1 `base/collection.tsv`

这是 round 1 的基础集合，用于第一次全量建索引。

### 6.2 `delta/roundX/collection.tsv`

这是第 X 轮新增文档，用于追加到现有索引，而不是重建全库。

### 6.3 `base/collection_meta.jsonl` 与 `delta/*/collection_meta.jsonl`

这里存放路由用上下文：

- `time_bucket_id`
- `source_id`
- `recency_norm`
- `cluster_label`
- `cluster_confidence`
- `cord_uid`
- `source_name`
- `publish_time`
- `first_seen_round`

对你这个 MoE 版本来说，真正关键的是前三个：

- `time_bucket_id`
- `source_id`
- `recency_norm`

因为它们是**在线可稳定获得**的字段，适合用于索引阶段和增量插入阶段的路由一致性。

### 6.4 `rounds/roundX/queries.tsv` 与 `qrels.tsv`

这是每一轮评测时用的查询和标注文件。

---

## 7. 原始 ColBERTv1 baseline 怎么做

这部分不要启用 MoE，不要启用 LoRA，不要启用 router。

### 7.1 训练

```bash
python -m colbert.train \
  --amp \
  --doc_maxlen 180 \
  --query_maxlen 32 \
  --dim 128 \
  --similarity l2 \
  --mask-punctuation \
  --bsize 16 \
  --accum 2 \
  --triples data/raw/msmarco/triples.train.jsonl \
  --queries data/raw/msmarco/queries.train.tsv \
  --collection data/raw/msmarco/collection.tsv \
  --root runs \
  --experiment msmarco_colbertv1 \
  --run base_l2 \
  --lr 3e-6 \
  --maxsteps 200000
```

单卡 4090 的建议：

- `bsize=16`
- `accum=2` 或 `4`
- `doc_maxlen=180`
- `query_maxlen=32`

### 7.2 在 TREC-COVID base 上建索引

```bash
python -m colbert.index \
  --amp \
  --doc_maxlen 180 \
  --query_maxlen 32 \
  --dim 128 \
  --similarity l2 \
  --mask-punctuation \
  --bsize 128 \
  --checkpoint runs/msmarco_colbertv1/train.py/base_l2/checkpoints/colbert.dnn \
  --collection data/processed/trec_covid/base/collection.tsv \
  --collection_meta data/processed/trec_covid/base/collection_meta.jsonl \
  --index_root indexes \
  --index_name trec_covid_base
```

### 7.3 构建 FAISS

```bash
python -m colbert.index_faiss \
  --index_root indexes \
  --index_name trec_covid_base \
  --partitions 2048 \
  --sample 0.3
```

### 7.4 追加 round2 delta

```bash
python -m colbert.index_delta \
  --amp \
  --doc_maxlen 180 \
  --query_maxlen 32 \
  --dim 128 \
  --similarity l2 \
  --mask-punctuation \
  --bsize 128 \
  --checkpoint runs/msmarco_colbertv1/train.py/base_l2/checkpoints/colbert.dnn \
  --collection data/processed/trec_covid/delta/round2/collection.tsv \
  --collection_meta data/processed/trec_covid/delta/round2/collection_meta.jsonl \
  --index_root indexes \
  --index_name trec_covid_base \
  --partitions 512 \
  --delta-name round2
```

### 7.5 检索并打分

```bash
python -m colbert.retrieve \
  --amp \
  --doc_maxlen 180 \
  --query_maxlen 32 \
  --dim 128 \
  --similarity l2 \
  --mask-punctuation \
  --queries data/processed/trec_covid/rounds/round2/queries.tsv \
  --checkpoint runs/msmarco_colbertv1/train.py/base_l2/checkpoints/colbert.dnn \
  --index_root indexes \
  --index_name trec_covid_base \
  --nprobe 32 \
  --partitions 2048 \
  --faiss_depth 1024 \
  --depth 1000
```

```bash
python scripts/score_ranking.py \
  --ranking runs/trec_covid_eval/ranking.tsv \
  --qrels data/processed/trec_covid/rounds/round2/qrels.tsv
```

---

## 8. 你的改造版模型怎么用这套数据

你的改造版重点不是替换掉 baseline，而是在**同一套 TREC-COVID base/delta 数据上**比较：

1. 原始 ColBERTv1
2. Frozen + 单 LoRA
3. Frozen + MoE LoRA
4. Frozen + MoE LoRA + routing context
5. Frozen + MoE LoRA + routing context + balance loss

这样实验才干净。

其中 routing context 建议只保留在线稳定字段：

- `time_bucket_id`
- `source_id`
- `recency_norm`

这意味着你在增量插入一个新文档时，不需要重新计算全局 drift descriptor，也不需要重新聚类全库。

---

## 9. 单卡 4090 上最稳的实验节奏

建议按下面顺序推进：

### 阶段 A：把原始 baseline 先跑通

目标：

- MS MARCO 训练出一个可用 checkpoint
- TREC-COVID base 建索引成功
- 能完成 round-by-round 检索与打分

### 阶段 B：先做最简 MoE 版

目标：

- 文档侧两遍前向
- top-1 expert routing
- query 侧保持共享

### 阶段 C：再加 balance loss

目标：

- 观察是否发生 routing collapse
- 分析不同 round 的专家分配比例

### 阶段 D：再做增量效果分析

目标：

- round 1 训练 / 初始化
- round 2~5 逐轮增量索引
- 看不同 round 的 MRR / Recall / expert usage 变化

---

## 10. 论文里最该写清楚的点

你这篇论文真正的贡献点，不是“用了 LoRA”，而是：

1. **在 ColBERT 词向量级离线索引框架下，实现了可增量的文档侧路由适配；**
2. **路由输入只依赖在线稳定可得的上下文字段，而不依赖离线全局重计算；**
3. **训练时路由、索引时路由、检索时路由保持一致；**
4. **在 evolving collection 上优于静态单适配器方案。**

如果这四点写清楚，你这篇文章才会更像“面向漂移检索的系统性方案”，而不是简单的参数高效微调拼装。

---

## 11. 我已经帮你补好的部分

这次我已经完成的和数据集直接相关的工作是：

- 新增并整理了 TREC-COVID 增量处理脚本
- 新增轻量 BM25 初排脚本
- 新增 qrels 标注脚本
- 新增 ranking 打分脚本
- 增补 `docids_mode`，避免 round 语义误判
- 补了 4090 友好的数据管线文档
- 用 toy 数据实际跑通了 `prepare -> bm25 -> annotate -> score` 链路

---

## 12. 下一步你该做什么

你现在最合理的动作不是继续改模型，而是先准备两份原始数据：

1. `MS MARCO Passage Ranking`
2. `CORD-19 + TREC-COVID rounds/qrels`

拿到原始文件后，先跑：

```bash
python scripts/prepare_trec_covid_incremental.py ...
```

等你把原始文件放到本地后，我下一步可以继续直接帮你做两件事：

1. 给你写一份**从原始数据到 baseline 结果**的逐条命令清单；
2. 再给你补一份**论文实验表格模板**，包括 baseline、ablation、incremental round-by-round 对比。
