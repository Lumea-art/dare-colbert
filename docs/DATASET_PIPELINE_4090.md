# 4090-Friendly Dataset Pipeline

This project now includes a minimal dataset pipeline for a paper-friendly setup:

- **Training / static baseline**: MS MARCO Passage Ranking
- **Incremental drift evaluation**: TREC-COVID over cumulative CORD-19 rounds

The rationale is simple:

- MS MARCO gives a strong and stable ColBERTv1 supervised baseline.
- TREC-COVID gives a true evolving collection with round-wise growth, which matches the new base/delta indexing flow.
- Both can run on a single RTX 4090 if you keep the model frozen on the document side and avoid full-domain retraining from scratch.

## 1. Recommended Protocol

### Static Baseline

1. Train original ColBERTv1 on MS MARCO.
2. Validate with `colbert.test`.
3. Build a static end-to-end index.

### Incremental Evaluation

1. Prepare TREC-COVID round 1 as the **base** collection.
2. Prepare rounds 2-5 as **delta** segments.
3. Build the base index.
4. Append each delta round with `colbert.index_delta`.
5. Retrieve after each round and score with `scripts/score_ranking.py`.

This keeps the experiment realistic and small enough for a single 4090.

## 2. TREC-COVID Preparation

The script `scripts/prepare_trec_covid_incremental.py` expects:

- `metadata.csv` from CORD-19
- `docids-rnd{round}.txt`
- `topics-rnd{round}.xml` or `.txt`
- `qrels-covid_d5_j0.5-rnd{round}.txt`

Example:

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

Main outputs:

- `data/processed/trec_covid/base/collection.tsv`
- `data/processed/trec_covid/base/collection_meta.jsonl`
- `data/processed/trec_covid/delta/round2/...`
- `data/processed/trec_covid/delta/round3/...`
- `data/processed/trec_covid/rounds/round1/queries.tsv`
- `data/processed/trec_covid/rounds/round1/qrels.tsv`
- `data/processed/trec_covid/manifest.json`

This script treats each paper as a single ColBERT document using `title + abstract` by default. That is deliberate: it is much cheaper and is usually the right first paper setup on a 4090.

`--docids-mode cumulative` means each `docids-rndX.txt` already contains the full collection visible at round `X`.
If your raw files instead contain only the newly added documents per round, switch to `--docids-mode incremental`.
On PowerShell, quote pattern arguments such as `'topics-rnd{round}.xml'` and `'qrels-rnd{round}.txt'`, otherwise `{round}` may be swallowed by the shell.

## 2.5 Build Static Full-Rebuild Rounds

If you want a strict original-`ColBERTv1` static baseline, build one full cumulative collection per round:

```bash
python scripts/build_static_rounds.py \
  --incremental-root data/processed/trec_covid \
  --output-dir data/processed/trec_covid_static
```

This creates:

- `data/processed/trec_covid_static/round1/collection.tsv`
- `data/processed/trec_covid_static/round2/collection.tsv`
- `data/processed/trec_covid_static/round3/collection.tsv`
- ...

Each `roundX` directory also includes:

- `queries.tsv`
- `qrels.tsv`
- `collection_meta.jsonl` when metadata exists in the incremental source

These round directories are intended for the `ColBERTv1-Static-FullRebuild` baseline, where each round is indexed from scratch as one monolithic collection.

## 2.6 Build a Smaller 4090-Friendly Subset

If you want a smaller but still valid incremental benchmark, build a reduced dataset from the processed `base + delta` root:

```bash
python scripts/build_small_trec_covid.py \
  --input-root data/processed/trec_covid \
  --output-dir data/processed/trec_covid_small \
  --rounds 1 2 3 \
  --max-base-docs 25000 \
  --max-delta-docs 8000 \
  --seed 13
```

This script:

- keeps all qrel-positive documents needed by the retained queries
- samples extra non-qrel documents up to the requested per-segment budget
- reassigns document PIDs to stay contiguous and safe for ColBERT training / indexing
- rewrites `collection.tsv`, `collection_meta.jsonl`, `queries.tsv`, `qrels.tsv`, `manifest.json`, and `pid_lookup.jsonl`

Recommended first paper setting on a single `4090`:

- `rounds 1 2 3`
- `max-base-docs 25000`
- `max-delta-docs 8000`
- keep all official queries unless you only need a quick smoke run

If you want an even faster debug set, also add:

```bash
--max-queries-per-round 10
```

## 3. Lightweight BM25 Baseline

For TREC-COVID scale, you can use the included pure-Python BM25 script:

```bash
python scripts/simple_bm25_topk.py \
  --collection data/processed/trec_covid/base/collection.tsv \
  --queries data/processed/trec_covid/rounds/round1/queries.tsv \
  --output runs/trec_round1.bm25.tsv \
  --topk 1000
```

This is not a replacement for Pyserini on very large corpora like full MS MARCO, but it is sufficient for a moderate, paper-friendly drift benchmark.

## 4. Annotating Ranked Lists for Triple Sampling

If you need hard negatives for a smaller supervised experiment:

```bash
python scripts/annotate_ranking_with_qrels.py \
  --ranking runs/trec_round1.bm25.tsv \
  --qrels data/processed/trec_covid/rounds/round1/qrels.tsv \
  --output runs/trec_round1.bm25.annotated.tsv
```

Then generate triples with the existing utility:

```bash
python utility/supervision/triples.py \
  --ranking runs/trec_round1.bm25.annotated.tsv \
  --output data/processed/trec_covid/train.triples.jsonl \
  --positives 1,10 \
  --depth 200 \
  --biased 100
```

## 5. Baseline Commands

### Original ColBERTv1 on MS MARCO

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
  --triples data/msmarco/train.triples.jsonl \
  --queries data/msmarco/queries.train.tsv \
  --collection data/msmarco/collection.tsv \
  --root runs \
  --experiment msmarco_colbertv1 \
  --run base_l2 \
  --lr 3e-6 \
  --maxsteps 200000
```

### Base Index for TREC-COVID

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

```bash
python -m colbert.index_faiss \
  --index_root indexes \
  --index_name trec_covid_base \
  --partitions 2048 \
  --sample 0.3
```

### Append a Delta Round

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

### Retrieve and Score

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

## 6. Practical Notes

- For a single 4090, prefer `bsize=16` or `bsize=32` with accumulation during training.
- For the MoE document-side model, freeze the BERT base and start with `num_experts=3`, `lora_rank=8`.
- For the first paper version, keep TREC-COVID as `title + abstract`; full-body passage expansion is optional and significantly more expensive.
- For the original ColBERTv1 baseline on MS MARCO, do **not** enable the MoE-specific flags.
