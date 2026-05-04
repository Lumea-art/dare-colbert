import torch
from colbert.utils.metadata import collate_metadata, default_doc_meta


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize,
                      positive_metadata=None, negative_metadata=None):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    positive_metadata = positive_metadata or [default_doc_meta() for _ in range(N)]
    negative_metadata = negative_metadata or [default_doc_meta() for _ in range(N)]

    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]
    positive_metadata = [positive_metadata[idx] for idx in indices.tolist()]
    negative_metadata = [negative_metadata[idx] for idx in indices.tolist()]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    batches = []
    for batch_idx, ((q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask)) in enumerate(zip(query_batches, positive_batches, negative_batches)):
        offset = batch_idx * bsize
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        D_meta = collate_metadata(
            positive_metadata[offset:offset+bsize] +
            negative_metadata[offset:offset+bsize]
        )
        batches.append((Q, D, D_meta))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches
