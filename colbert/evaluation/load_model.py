import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint


def load_model(args, do_print=True):
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
    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint
