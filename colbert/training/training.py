import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
from colbert.utils.metadata import IGNORE_CLUSTER_LABEL


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation,
                                      enable_moe_doc=args.enable_moe_doc,
                                      num_experts=args.num_experts,
                                      lora_rank=args.lora_rank,
                                      lora_alpha=args.lora_alpha,
                                      lora_dropout=args.lora_dropout,
                                      freeze_bert=args.freeze_bert,
                                      freeze_linear=args.freeze_linear,
                                      router_hidden_size=args.router_hidden_size,
                                      router_temp=args.router_temp,
                                      num_time_buckets=args.num_time_buckets,
                                      time_bucket_dim=args.time_bucket_dim,
                                      num_source_buckets=args.num_source_buckets,
                                      source_dim=args.source_dim,
                                      prototype_path=args.prototype_path)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0

        for queries, passages, doc_metadata in BatchSteps:
            with amp.context():
                outputs = colbert(queries, passages, doc_metadata=doc_metadata, return_details=True)
                scores = outputs['scores'].view(2, -1).permute(1, 0)
                contrastive_loss = criterion(scores, labels[:scores.size(0)])
                balance_loss = compute_balance_loss(outputs.get('router_probs'), outputs.get('expert_ids'))
                router_loss = compute_router_supervision_loss(outputs.get('router_logits'), doc_metadata)

                balance_weight = warmup_weight(args.balance_alpha, args.balance_warmup, batch_idx)
                router_weight = warmup_weight(args.router_supervision_alpha, args.router_supervision_warmup, batch_idx)

                loss = contrastive_loss + balance_weight * balance_loss + router_weight * router_loss
                loss = loss / args.accumsteps

            if args.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            print_message(batch_idx, avg_loss)
            manage_checkpoints(args, colbert, optimizer, batch_idx+1)


def warmup_weight(alpha, warmup_steps, step_idx):
    if alpha <= 0.0:
        return 0.0

    if warmup_steps <= 0:
        return alpha

    return alpha * min(1.0, float(step_idx + 1) / float(warmup_steps))


def compute_balance_loss(router_probs, expert_ids):
    if router_probs is None or expert_ids is None:
        return torch.tensor(0.0, device=DEVICE)

    num_experts = router_probs.size(-1)
    expert_fraction = F.one_hot(expert_ids, num_classes=num_experts).float().mean(dim=0)
    router_fraction = router_probs.mean(dim=0)
    return num_experts * torch.sum(expert_fraction * router_fraction)


def compute_router_supervision_loss(router_logits, doc_metadata):
    if router_logits is None or doc_metadata is None:
        return torch.tensor(0.0, device=DEVICE)

    if 'cluster_labels' not in doc_metadata:
        return torch.tensor(0.0, device=DEVICE)

    cluster_labels = doc_metadata['cluster_labels'].to(device=router_logits.device, dtype=torch.long)
    confidence = doc_metadata.get('cluster_confidence')
    if confidence is None:
        confidence = torch.ones_like(cluster_labels, dtype=router_logits.dtype, device=router_logits.device)
    else:
        confidence = confidence.to(device=router_logits.device, dtype=router_logits.dtype)

    valid = (cluster_labels != IGNORE_CLUSTER_LABEL) & (cluster_labels >= 0) & (cluster_labels < router_logits.size(-1))
    if not torch.any(valid):
        return torch.tensor(0.0, device=router_logits.device)

    losses = F.cross_entropy(router_logits[valid], cluster_labels[valid], reduction='none')
    return (losses * confidence[valid]).mean()
