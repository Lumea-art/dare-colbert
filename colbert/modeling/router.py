import torch
import torch.nn as nn
import torch.nn.functional as F


class DriftRouter(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_experts,
                 hidden_dim=256,
                 num_time_buckets=0,
                 time_bucket_dim=8,
                 num_source_buckets=0,
                 source_dim=8,
                 prototype_dim=0):
        super().__init__()

        self.num_time_buckets = num_time_buckets
        self.num_source_buckets = num_source_buckets
        self.prototype_dim = prototype_dim

        self.time_embeddings = nn.Embedding(num_time_buckets, time_bucket_dim) if num_time_buckets > 0 else None
        self.source_embeddings = nn.Embedding(num_source_buckets, source_dim) if num_source_buckets > 0 else None

        input_dim = hidden_size
        input_dim += time_bucket_dim if self.time_embeddings is not None else 0
        input_dim += source_dim if self.source_embeddings is not None else 0
        input_dim += 1
        input_dim += prototype_dim

        self.layer_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_dim, num_experts)

    def forward(self, doc_repr, metadata=None, proto_scores=None):
        metadata = metadata or {}
        pieces = [doc_repr]
        batch_size = doc_repr.size(0)
        device = doc_repr.device

        if self.time_embeddings is not None:
            time_bucket_ids = metadata.get('time_bucket_ids')
            if time_bucket_ids is None:
                time_bucket_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                time_bucket_ids = time_bucket_ids.to(device=device, dtype=torch.long)
            time_bucket_ids = time_bucket_ids.clamp(min=0, max=self.num_time_buckets - 1)
            pieces.append(self.time_embeddings(time_bucket_ids))

        if self.source_embeddings is not None:
            source_ids = metadata.get('source_ids')
            if source_ids is None:
                source_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                source_ids = source_ids.to(device=device, dtype=torch.long)
            source_ids = source_ids.clamp(min=0, max=self.num_source_buckets - 1)
            pieces.append(self.source_embeddings(source_ids))

        recency_norm = metadata.get('recency_norm')
        if recency_norm is None:
            recency_norm = torch.zeros(batch_size, 1, dtype=doc_repr.dtype, device=device)
        else:
            recency_norm = recency_norm.to(device=device, dtype=doc_repr.dtype).view(batch_size, 1)
        pieces.append(recency_norm)

        if proto_scores is None:
            proto_scores = doc_repr.new_zeros((batch_size, self.prototype_dim))
        else:
            proto_scores = proto_scores.to(device=device, dtype=doc_repr.dtype)
        pieces.append(proto_scores)

        router_input = torch.cat(pieces, dim=-1)
        hidden = self.act(self.proj(self.layer_norm(router_input)))
        return self.out(hidden)


def straight_through_top1(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    expert_ids = probs.argmax(dim=-1)
    hard_assignments = F.one_hot(expert_ids, num_classes=probs.size(-1)).type_as(probs)
    gate_weights = hard_assignments + probs - probs.detach()
    return probs, gate_weights, expert_ids
