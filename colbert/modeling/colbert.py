import os
import string
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE
from colbert.modeling.lora import inject_expert_lora, iter_expert_lora_modules
from colbert.modeling.router import DriftRouter, straight_through_top1


class ColBERT(BertPreTrainedModel):
    def __init__(self, config,
                 query_maxlen,
                 doc_maxlen,
                 mask_punctuation,
                 dim=128,
                 similarity_metric='cosine',
                 enable_moe_doc=False,
                 num_experts=1,
                 lora_rank=0,
                 lora_alpha=16.0,
                 lora_dropout=0.0,
                 freeze_bert=False,
                 freeze_linear=False,
                 router_hidden_size=256,
                 router_temp=1.0,
                 num_time_buckets=0,
                 time_bucket_dim=8,
                 num_source_buckets=0,
                 source_dim=8,
                 prototype_path=None):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim
        self.enable_moe_doc = enable_moe_doc
        self.num_experts = num_experts
        self.router_temp = router_temp
        self.freeze_bert = freeze_bert

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        prototype_vectors = self._load_prototype_vectors(prototype_path, config.hidden_size)
        self.register_buffer('prototype_vectors', prototype_vectors)

        self.router = None
        self.expert_lora_modules = []
        if self.enable_moe_doc and self.num_experts > 0:
            inject_expert_lora(
                self.bert,
                target_suffixes=['query', 'value'],
                num_experts=self.num_experts,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            self.expert_lora_modules = list(iter_expert_lora_modules(self.bert))
            self.router = DriftRouter(
                hidden_size=config.hidden_size,
                num_experts=self.num_experts,
                hidden_dim=router_hidden_size,
                num_time_buckets=num_time_buckets,
                time_bucket_dim=time_bucket_dim,
                num_source_buckets=num_source_buckets,
                source_dim=source_dim,
                prototype_dim=self.prototype_vectors.size(0),
            )

        self.init_weights()

        if freeze_bert:
            for name, parameter in self.bert.named_parameters():
                parameter.requires_grad = ('lora_A' in name) or ('lora_B' in name)

        if freeze_linear:
            for parameter in self.linear.parameters():
                parameter.requires_grad = False

    def forward(self, Q, D, doc_metadata=None, return_details=False):
        query_embeddings = self.query(*Q)
        doc_outputs = self.doc(*D, metadata=doc_metadata, keep_dims=True, return_details=return_details)
        doc_embeddings = doc_outputs['embeddings'] if return_details else doc_outputs
        scores = self.score(query_embeddings, doc_embeddings)

        if not return_details:
            return scores

        doc_outputs['scores'] = scores
        return doc_outputs

    def query(self, input_ids, attention_mask):
        self._clear_doc_gates()
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True, metadata=None, return_details=False):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        metadata = self._move_metadata_to_device(metadata)

        router_details = {
            'doc_repr': None,
            'proto_scores': None,
            'router_logits': None,
            'router_probs': None,
            'expert_ids': None,
        }

        if self.enable_moe_doc and self.router is not None:
            shared_hidden = self._encode_shared_doc(input_ids, attention_mask)
            doc_repr = self._compute_doc_repr(shared_hidden, input_ids, attention_mask)
            proto_scores = self._compute_proto_scores(doc_repr)
            router_logits = self.router(doc_repr, metadata=metadata, proto_scores=proto_scores)
            router_probs, gate_weights, expert_ids = straight_through_top1(router_logits, self.router_temp)

            self._set_doc_gates(gate_weights)
            D = self.bert(input_ids, attention_mask=attention_mask)[0]
            self._clear_doc_gates()

            router_details = {
                'doc_repr': doc_repr,
                'proto_scores': proto_scores,
                'router_logits': router_logits,
                'router_probs': router_probs,
                'expert_ids': expert_ids,
            }
        else:
            self._clear_doc_gates()
            D = self.bert(input_ids, attention_mask=attention_mask)[0]

        D = self.linear(D)

        mask = self._doc_mask(input_ids, attention_mask).unsqueeze(2)
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        if not return_details:
            return D

        router_details['embeddings'] = D
        return router_details

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def _load_prototype_vectors(self, prototype_path, hidden_size):
        if prototype_path is None or not os.path.exists(prototype_path):
            return torch.empty(0, hidden_size)

        prototype_vectors = torch.load(prototype_path, map_location='cpu')
        if isinstance(prototype_vectors, dict):
            prototype_vectors = prototype_vectors.get('prototypes', prototype_vectors.get('vectors'))

        if prototype_vectors is None:
            return torch.empty(0, hidden_size)

        if prototype_vectors.dim() != 2 or prototype_vectors.size(1) != hidden_size:
            raise ValueError("Prototype vectors must have shape [num_prototypes, hidden_size].")

        return prototype_vectors.float()

    def _move_metadata_to_device(self, metadata):
        if metadata is None:
            return None

        moved = {}
        for key, value in metadata.items():
            moved[key] = value.to(DEVICE) if torch.is_tensor(value) else value
        return moved

    def _set_doc_gates(self, gate_weights):
        for module in self.expert_lora_modules:
            module.set_gate_weights(gate_weights)

    def _clear_doc_gates(self):
        for module in self.expert_lora_modules:
            module.clear_gate_weights()

    def _encode_shared_doc(self, input_ids, attention_mask):
        self._clear_doc_gates()
        context = torch.no_grad() if self.freeze_bert else nullcontext()
        with context:
            return self.bert(input_ids, attention_mask=attention_mask)[0]

    def _doc_mask(self, input_ids, attention_mask):
        mask = attention_mask.float()
        if self.mask_punctuation:
            mask = mask * torch.tensor(self.mask(input_ids), device=DEVICE).float()
        return mask

    def _compute_doc_repr(self, hidden_states, input_ids, attention_mask):
        mask = self._doc_mask(input_ids, attention_mask).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _compute_proto_scores(self, doc_repr):
        if self.prototype_vectors.size(0) == 0:
            return doc_repr.new_zeros((doc_repr.size(0), 0))

        normalized_doc_repr = F.normalize(doc_repr, p=2, dim=-1)
        normalized_prototypes = F.normalize(self.prototype_vectors.to(doc_repr.device), p=2, dim=-1)
        return normalized_doc_repr @ normalized_prototypes.transpose(0, 1)
