import ujson
import torch


IGNORE_CLUSTER_LABEL = -100


def default_doc_meta(pid=None):
    return {
        'pid': -1 if pid is None else int(pid),
        'time_bucket_id': 0,
        'source_id': 0,
        'recency_norm': 0.0,
        'cluster_label': IGNORE_CLUSTER_LABEL,
        'cluster_confidence': 1.0,
    }


def load_collection_meta(path, expected_size=None):
    if expected_size is not None:
        metadata = [default_doc_meta(pid) for pid in range(expected_size)]
    else:
        metadata = []

    if path is None:
        return metadata

    with open(path, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            if not line.strip():
                continue

            item = ujson.loads(line)
            pid = int(item['pid'])

            while len(metadata) <= pid:
                metadata.append(default_doc_meta(len(metadata)))

            merged = default_doc_meta(pid)
            merged.update({
                'time_bucket_id': int(item.get('time_bucket_id', 0)),
                'source_id': int(item.get('source_id', 0)),
                'recency_norm': float(item.get('recency_norm', 0.0)),
                'cluster_label': int(item.get('cluster_label', IGNORE_CLUSTER_LABEL)),
                'cluster_confidence': float(item.get('cluster_confidence', 1.0)),
            })
            metadata[pid] = merged

    return metadata


def collate_metadata(metadata_items):
    metadata_items = metadata_items or []
    if len(metadata_items) == 0:
        return {
            'pids': torch.empty(0, dtype=torch.long),
            'time_bucket_ids': torch.empty(0, dtype=torch.long),
            'source_ids': torch.empty(0, dtype=torch.long),
            'recency_norm': torch.empty(0, dtype=torch.float),
            'cluster_labels': torch.empty(0, dtype=torch.long),
            'cluster_confidence': torch.empty(0, dtype=torch.float),
        }

    return {
        'pids': torch.tensor([item.get('pid', -1) for item in metadata_items], dtype=torch.long),
        'time_bucket_ids': torch.tensor([item.get('time_bucket_id', 0) for item in metadata_items], dtype=torch.long),
        'source_ids': torch.tensor([item.get('source_id', 0) for item in metadata_items], dtype=torch.long),
        'recency_norm': torch.tensor([item.get('recency_norm', 0.0) for item in metadata_items], dtype=torch.float),
        'cluster_labels': torch.tensor([item.get('cluster_label', IGNORE_CLUSTER_LABEL) for item in metadata_items], dtype=torch.long),
        'cluster_confidence': torch.tensor([item.get('cluster_confidence', 1.0) for item in metadata_items], dtype=torch.float),
    }
